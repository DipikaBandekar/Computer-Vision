
class pca : public Classifier {
   public:
	pca(const vector<string> &_class_list) : Classifier(_class_list)
	{}
	
	virtual void train(const Dataset &filenames) 
	{
           for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
	     {
		cout << "Processing " << c_iter->first << endl;
		CImg<double> class_vectors(size*size*3, c_iter->second.size(), 1);

		// convert each image to be a row of this "model" image
		for(int i=0; i<c_iter->second.size(); i++)
		  class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));
		cout<<"class_vectors.rows() and .cols()"<<class_vectors.height()<<"  "<<class_vectors.width()<<endl;
		class_vectors.save_png(("pca_model." + c_iter->first + ".png").c_str());
	      }		
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

    static const int size = 40; // subsampled image resolution
    map<string, CImg<double> > models; // trained models

};

