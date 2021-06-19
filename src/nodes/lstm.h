/* This file is part of onnx2c.
 *
 * LSTM.
 * Implements a Long Short Term Memory node.
 * A nice description is given in:
 * http://colah.github.io/posts/2015-08-Understanding-LSTMs/
 * A nice video with the maths written out:
 * https://www.youtube.com/watch?v=Opj2AT0iYCw
 * The exact equations used for ONNX's LSTM are given in the
 * specification, and the two seem to match the two links above.
 *
 * NB: Y_h and Y_c must both always be available, even if network
 * marks them optional. (Since they are the recursion tensors).
 * If the initializers initial_h or initial_c are given, the Y_[h,c]
 * tensors are aliased to the respectie one, and the LSTM hidden/cell
 * state is saved & updated in the initial_[h,c] tensor.
 */
namespace toC {

class LSTM : public Node {
	public:
	LSTM() {
		op_name = "LSTM";
		clip = -1.0;
		hidden_size = -1;
		input_forget = 0;
		X=NULL;
		W=NULL;
		R=NULL;
		B=NULL;
		sequence_lens=NULL;
		initial_h=NULL;
		initial_c=NULL;
		P=NULL;
		Y=NULL;
		Y_h=NULL;
		Y_c=NULL;
	}

	// inputs
	const Tensor *X;
	const Tensor *W;
	const Tensor *R;
	// optional inputs
	const Tensor *B;
	const Tensor *sequence_lens;
	const Tensor *initial_h;
	const Tensor *initial_c;
	const Tensor *P;
	// optional outputs
	Tensor *Y;
	Tensor *Y_h;
	Tensor *Y_c;

	// Attributes
	std::vector<float> activation_alpha;
	std::vector<float> activation_beta;
	std::vector<std::string> activations; // in order, activations f, g, & h
	float clip;  // negative for no clip
	std::string direction;
	int hidden_size;
	int input_forget;


	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		X->print_tensor(dst, !decorate, !decorate?"":"X");
		dst << ", ";
		W->print_tensor(dst, !decorate, !decorate?"":"W");
		dst << ", ";
		R->print_tensor(dst, !decorate, !decorate?"":"R");
		if( B ) {
			dst << ", ";
			B->print_tensor(dst, !decorate, !decorate?"":"B");
		}
		if( sequence_lens ) {
			dst << ", ";
			sequence_lens->print_tensor(dst, !decorate, !decorate?"":"sequence_lens");
		}
		if( initial_h ) {
			dst << ", ";
			initial_h->print_tensor(dst, !decorate, !decorate?"":"Y_h");
		}
		if( initial_c ) {
			dst << ", ";
			initial_c->print_tensor(dst, !decorate, !decorate?"":"Y_c");
		}
		if( P ) {
			dst << ", ";
			P->print_tensor(dst, !decorate, !decorate?"":"P");
		}
		if( Y->is_used() ) {
			dst << ", ";
			Y->print_tensor(dst, !decorate, !decorate?"":"Y");
		}
		if( Y_h->is_used() && Y_h->isAliasOf==NULL ) {
			dst << ", ";
			Y_h->print_tensor(dst, !decorate, !decorate?"":"Y_h");
		}
		if( Y_c->is_used() && Y_c->isAliasOf==NULL) {
			dst << ", ";
			Y_c->print_tensor(dst, !decorate, !decorate?"":"Y_c");
		}
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;

			if( a.name() == "activation_alpha" )
				activation_alpha = parse_attribute_floats(a);
			else if( a.name() == "activation_beta" )
				activation_beta = parse_attribute_floats(a);
			else if( a.name() == "activations" )
				activations = parse_attribute_strings(a);
			else if( a.name() == "clip" )
				clip = parse_attribute_float(a);
			else if( a.name() == "direction" ) {
				direction = parse_attribute_string(a);
				if( direction == "" ) direction="forward";
				else if(  direction != "forward"
				        &&direction != "reverse"
				        &&direction != "bidirectional")
					ERROR("Bad value ("<<direction<<") for direction attribute");
			}
			else if( a.name() == "hidden_size" )
				hidden_size = parse_attribute_int(a);
			else if( a.name() == "input_forget" )
				input_forget = parse_attribute_int(a);
			else
				ERROR("Bad attribute " << a.name() << " for LSTM");
		}
	}

	float get_activation_alpha( const std::string &a)
	{
		/* These activations don't have an alpha */
		if( a == "Sigmoid" )
			return 0;
		if( a == "Tanh" )
			return 0;
		if( a == "Relu" )
			return 0;

		ERROR("Unhandled: alpha for activation: " << a);
	}
	float get_activation_beta( const std::string &a)
	{
		/* These activations don't have a beta */
		if( a == "Sigmoid" )
			return 0;
		if( a == "Tanh" )
			return 0;
		if( a == "Relu" )
			return 0;

		ERROR("Unhandled: beta for activation: " << a);
	}

	void print_activation(std::ostream &dst, const std::string &activation, const std::string &variable) const
	{
		if( activation == "Sigmoid" )
			dst << "1.0f/(1+expf(-" << variable << "));" << std::endl;
		else if (activation == "Tanh" )
			// TODO: optimize to tanhf? If someone uses tanh, do they care about execution speed? :)
			dst << "tanh(" << variable << ");" << std::endl;
		else if (activation == "Relu" )
			dst << "MAX(" << variable << ", 0);" << std::endl;
		else
			ERROR("Unimplmemented activation function");
	}


	/* Print the C code for the core LSTM kernel, inside of the "sequences" loop.
	 * The code is almost identical for forward and backwards nodes */
	void print_lstm_kernel(std::ostream &dst, bool forward) const
	{
		int dir;    // direction index into tensors that separate forward and backward (W,B,Y,...)  
		int f_act;  // indexes for the activation functions in activations[]
		int g_act;
		int h_act;
		std::string di;
		if( forward ) {
			dir=0;
			f_act=0;
			g_act=1;
			h_act=2;
			di="k";
		}
		else {
			dir=1;
			f_act=3;
			g_act=4;
			h_act=5;
			di="ds-1-k";
		}

		dst << "\t" << "for( int i=0; i<bs; i++)" << std::endl;
		dst << "\t" << "for( int j=0; j<hs; j++) {" << std::endl;
		dst << "\t\t" << "ft[i][j]=0;" << std::endl;
		dst << "\t\t" << "it[i][j]=0;" << std::endl;
		dst << "\t\t" << "ct[i][j]=0;" << std::endl;

		// Xt*W
		dst << "\t\t" << "for( int k=0; k<ds; k++) {" << std::endl;
		dst << "\t\t\t" << "ft[i][j] += X[s][i]["<<di<<"]*W["<<dir<<"][fidx+j][k];" << std::endl;
		dst << "\t\t\t" << "it[i][j] += X[s][i]["<<di<<"]*W["<<dir<<"][iidx+j][k];" << std::endl;
		dst << "\t\t\t" << "ct[i][j] += X[s][i]["<<di<<"]*W["<<dir<<"][cidx+j][k];" << std::endl;
		dst << "\t\t" << "}" << std::endl;

		// Ht-1*R
		dst << "\t\t" << "for( int k=0; k<hs; k++) {" << std::endl;
		dst << "\t\t\t" << "ft[i][j] += Y_h["<<dir<<"][i][k]*R["<<dir<<"][fidx+j][k];" << std::endl;
		dst << "\t\t\t" << "ct[i][j] += Y_h["<<dir<<"][i][k]*R["<<dir<<"][cidx+j][k];" << std::endl;
		dst << "\t\t\t" << "it[i][j] += Y_h["<<dir<<"][i][k]*R["<<dir<<"][iidx+j][k];" << std::endl;
		dst << "\t\t" << "}" << std::endl;

		if( B ) { // Bias
		dst << "\t\t" << "ft[i][j] += B["<<dir<<"][fidx+j];" << std::endl;
		dst << "\t\t" << "ft[i][j] += B["<<dir<<"][Rb+fidx+j];" << std::endl;
		dst << "\t\t" << "it[i][j] += B["<<dir<<"][iidx+j];" << std::endl;
		dst << "\t\t" << "it[i][j] += B["<<dir<<"][Rb+iidx+j];" << std::endl;
		dst << "\t\t" << "ct[i][j] += B["<<dir<<"][cidx+j];" << std::endl;
		dst << "\t\t" << "ct[i][j] += B["<<dir<<"][Rb+cidx+j];" << std::endl;
		}
		if( P ) { // Peephole
		dst << "\t\t" << "ft[i][j] += P["<<dir<<"][fidx+j]*Y_c["<<dir<<"][i][j];" << std::endl;
		dst << "\t\t" << "it[i][j] += P["<<dir<<"][iidx+j]*Y_c["<<dir<<"][i][j];" << std::endl;
		// Cell gate does not have a peephole
		}

		// Activations
		dst << "\t\t" << "ft[i][j] =";
		print_activation( dst, activations[f_act], "ft[i][j]");
		dst << "\t\t" << "it[i][j] =";
		print_activation( dst, activations[f_act], "it[i][j]");
		dst << "\t\t" << "ct[i][j] =";
		print_activation( dst, activations[g_act], "ct[i][j]");
		dst << "\t" << "}" << std::endl;

		// Cell state, Output gate
		dst << "\t" << "for( int i=0; i<bs; i++)" << std::endl;
		dst << "\t" << "for( int j=0; j<hs; j++) {" << std::endl;
		dst << "\t\t" << "/* Cell state */" << std::endl;
		dst << "\t\t" << "Y_c["<<dir<<"][i][j] = Y_c["<<dir<<"][i][j]*ft[i][j] + it[i][j]*ct[i][j];" << std::endl;
		dst << "\t\t" << "/* Output gate */" << std::endl;
		dst << "\t\t" << "ot[i][j]=0;" << std::endl;
		// X*W
		dst << "\t\t" << "for( int k=0; k<ds; k++)" << std::endl;
		dst << "\t\t\t" << "ot[i][j] += X[s][i]["<<di<<"]*W["<<dir<<"][oidx+j][k];" << std::endl;
		// Ht-1*R
		dst << "\t\t" << "for( int k=0; k<hs; k++)" << std::endl;
		dst << "\t\t\t" << "ot[i][j] += Y_h["<<dir<<"][i][k]*R["<<dir<<"][oidx+j][k];" << std::endl;
		if( B ) {// Bias
		dst << "\t\t" << "ot[i][j] += B["<<dir<<"][oidx+j];" << std::endl;
		dst << "\t\t" << "ot[i][j] += B["<<dir<<"][Rb+oidx+j];" << std::endl;
		}
		if( P ) // Peephole
		dst << "\t\t" << "ot[i][j] += P["<<dir<<"][oidx+j]*Y_c["<<dir<<"][i][j];" << std::endl;
		dst << "\t\t" << "ot[i][j] =";
		print_activation( dst, activations[f_act], "ot[i][j]");
		dst << "\t" << "}" << std::endl;

		// Hidden state
		dst << "\t" << "/* Hidden state */" << std::endl;
		dst << "\t" << "for( int i=0; i<bs; i++)" << std::endl;
		dst << "\t" << "for( int j=0; j<hs; j++) {" << std::endl;
			dst << "\t\t" << "Y_h["<<dir<<"][i][j] = ot[i][j] * ";
				std::string activated="Y_c[" + std::to_string(dir) + "][i][j]"; 
				print_activation( dst, activations[h_act], activated );
			if( Y->is_used() ) {
				dst << "\t\t" << "Y[s]["<<dir<<"][i][j] = Y_h["<<dir<<"][i][j];" << std::endl;
			}
		dst << "\t" << "}" << std::endl << std::endl;

	}

	virtual void print(std::ostream &dst) const override
	{
		dst << "\t/* LSTM " << std::endl;
		dst << "\t * inputs: " << std::endl;
		dst << "\t *   X = " << X->cname() << std::endl;
		dst << "\t *   W = " << W->cname() << std::endl;
		dst << "\t *   R = " << R->cname() << std::endl;
		dst << "\t *   B = " << (B?B->cname():"") << std::endl;
		dst << "\t *   sequence_lens = " << (sequence_lens?sequence_lens->cname():"") << std::endl;
		dst << "\t *   initial_h = " << (initial_h?initial_h->cname():"") << std::endl;
		dst << "\t *   initial_c = " << (initial_c?initial_c->cname():"") << std::endl;
		dst << "\t *   P = " << (P?P->cname():"") << std::endl;
		dst << "\t * outputs: " << std::endl;
		dst << "\t *   Y = " << Y->cname() << std::endl;
		dst << "\t *   Y_h = " << Y_h->cname() << std::endl;
		dst << "\t *   Y_c = " << Y_c->cname() << std::endl;
		dst << "\t * attributes:" << std::endl;
		dst << "\t *   activations: ";
			for( auto a : activations )
				dst << a << " ";
			dst << std::endl;
		dst << "\t * (rest TBD):" << std::endl;
		dst << "\t */" << std::endl;

		const std::string data_type = X->data_type_str();

		int hs = R->data_dim[2]; //hidden size
		int ds = X->data_dim[2]; //input (data) size
		int bs = X->data_dim[1]; // batch size


		dst << "\t" << "int hs = " << hs << ";" << std::endl;
		dst << "\t" << "int ds = " << ds << ";" << std::endl;
		dst << "\t" << "int bs = " << bs << ";" << std::endl;
		// index into W, R to get the start of the gate indices
		dst << "\t" << "int iidx = 0;" << std::endl;
		dst << "\t" << "int oidx = hs;" << std::endl;
		dst << "\t" << "int fidx = 2*hs;" << std::endl;
		dst << "\t" << "int cidx = 3*hs;" << std::endl;
		// index into B, to get Rb. Add *iidx too. Wb is B at offset 0
		dst << "\t" << "int Rb = 4*hs;" << std::endl;
		// TODO: variable lenght sequences not yet implemented
		dst << "\t" << "int sequence_lenght = " <<  X->data_dim[0] << ";" << std::endl;

		// TODO: these temporary variables are BIG. Make them global to minimize
		// stack usage? Probably needs to be an onnx2c flag for user to select
		dst << "\t" << "/* Forget gate */" << std::endl;
		dst << "\t" << data_type << " ft[bs][hs];" << std::endl;
		dst << "\t" << "/* Input gate */" << std::endl;
		dst << "\t" << data_type << " it[bs][hs];" << std::endl;
		dst << "\t" << "/* Cell gate */" << std::endl;
		dst << "\t" << data_type << " ct[bs][hs];" << std::endl;
		dst << "\t" << "/* Output gate */" << std::endl;
		dst << "\t" << data_type << " ot[bs][hs];" << std::endl;
		dst << std::endl;
		dst << "\t" << "for( int s=0; s<sequence_lenght; s++) {" << std::endl;

		print_lstm_kernel(dst, /* forward= */ true); 
		if( direction == "bidirectional" )
			print_lstm_kernel(dst, /* forward= */ false); 

		dst << "\t" << "} /* sequences */" << std::endl;

	}

	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		if( inputs.size() < 3 || inputs.size() > 8 )
			ERROR("wrong number of inputs to LSTM");


		// Set attribute default values for those attributes that are not set in the model
		if( activations.size() == 0 ) {
			activations.push_back("Sigmoid");
			activations.push_back("Tanh");
			activations.push_back("Tanh");
			if( direction == "bidirectional" ) {
				activations.push_back("Sigmoid");
				activations.push_back("Tanh");
				activations.push_back("Tanh");
			}
		}
		if( activations.size() == 6 )
		if( activations.size() != 3 && activations.size() != 6)
			ERROR("Error - bad number of activations attributes");

		if( activation_alpha.size() == 0 ) {
			for( auto &a : activations ) {
				activation_alpha.push_back(get_activation_alpha(a));
			}
		}
		if( activation_alpha.size() != 3 && activation_alpha.size() != 6)
			ERROR("Unimplemented/error: not 3(6) activation alphas");

		if( activation_beta.size() == 0 ) {
			for( auto &a : activations ) {
				activation_beta.push_back(get_activation_beta(a));
			}
		}
		if( activation_beta.size() != 3 && activation_beta.size() != 6)
			ERROR("Unimplemented/error: not 3(6) activation betas");

		// 'reverse' calculates the same as 'forward', it is left to the caller
		// to reverse the input order
		//if( direction == "" || direction == "forward" || direction == "reverse" )
		//	// TODO: this variable should be "unidirectional"
		//	direction = "forward";
		//else
		//	ERROR("Unimplmeneted: backward and bidirectional LSTM");

		if( hidden_size < 0 )
			ERROR("Must provide hidden_size attribute!");

		X = inputs[0];
		W = inputs[1];
		R = inputs[2];
		//optional inputs. Trailing unprovided inputs can just be left out
		//but non-trailing, unprovided inputs MUST have an empty string as name
		// (guess that means tensors MAY NOT have an empty string as name?)
		if( inputs.size() > 3 && inputs[3]->name != "" )
			B = inputs[3];
		if( inputs.size() > 4 && inputs[4]->name != "" )
			sequence_lens = inputs[4];
		if( inputs.size() > 5 && inputs[5]->name != "" )
			initial_h = inputs[5];
		if( inputs.size() > 6 && inputs[6]->name != "" )
			initial_c = inputs[6];
		if( inputs.size() > 7 && inputs[7]->name != "" )
			P = inputs[7];


		int seq_length = X->data_dim[0];
		int batch_size = X->data_dim[1];
		int num_directions = W->data_dim[0];

		if( sequence_lens ) {
			if( static_cast<int>(sequence_lens->rank()) != batch_size )
				ERROR("If providing sequence lengths, there must be 'batch_size' of them");
			for( auto sl : sequence_lens->data_dim )
				if( sl < seq_length )
					// Not quite sure if I understand the documentation correctly here.
					ERROR("Error: requested sequence lenght is longer than input data");
		}


		// Generate output tensors.

		Y = new Tensor;
		Y->data_type = X->data_type;
		std::vector<int> y_size({ seq_length, num_directions, batch_size, hidden_size });
		Y->data_dim = y_size;

		// Y_h and Y_c are special: optional as outputs to the rest of the network,
		// but mandatory as outputs to this node itself. Also, they alias
		// on top of initial_[h,c], which is implemented copying initializers
		// for Y_[h,c].
		Y_h = new Tensor;
		Y_h->data_type = X->data_type;
		std::vector<int> yh_size({ num_directions, batch_size, hidden_size });
		Y_h->data_dim = yh_size;

		Y_h->isRecursive=true;
		if( initial_h ) {
			Y_h->isAliasOf = initial_h;
			Y_h->generate=false;
		}
		else {
			Y_h->data_buffer = calloc(Y_h->data_num_elem(), Y_h->data_elem_size());
			if( Y_h->data_buffer == NULL )
				ERROR("Memory allocation failed");
			Y_h->initialize = true;
		}

		Y_c = new Tensor;
		Y_c->data_type = X->data_type;
		std::vector<int> yc_size({ num_directions, batch_size, hidden_size });
		Y_c->data_dim = yc_size;

		Y_c->isRecursive=true;
		if( initial_c ) {
			Y_c->isAliasOf = initial_c;
			Y_c->generate=false;
		}
		else {
			Y_c->data_buffer = calloc(Y_c->data_num_elem(), Y_c->data_elem_size());
			if( Y_c->data_buffer == NULL )
				ERROR("Memory allocation failed");
			Y_c->initialize = true;
		}

		outputs.push_back(Y);
		outputs.push_back(Y_h);
		outputs.push_back(Y_c);
	}
};
}

