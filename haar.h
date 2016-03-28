class haar : public Classifier {
public:
	haar(const vector<string> &_class_list) : Classifier(_class_list)
	{}

	virtual void train()
	{

	}


   virtual void load_model() {
        cout << "empty load function" << endl;
    }

    virtual string classify(const string &filename) {
        cout << "empty classify function" << endl;
    }

protected:
	CImg<double> extract_features(const string &filename) 
	{
        	return (CImg<double>(filename.c_str())).resize(size, size, 1, 3).unroll('x');
    	}

    static const int size = 100; // subsampled image resolution
    map<string, CImg<double> > models; // trained models
   };