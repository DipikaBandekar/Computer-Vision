
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
			
			class_vectors = subtract_mean(class_vectors);
			CImg<double> mul = covariate(class_vectors);
			cout<<"rows and cols of matrix mux: "<<mul.width()<<"   "<<mul.height()<<endl;
			//cout<<"class_vectors.rows() and .cols()"<<class_vectors.height()<<"  "<<class_vectors.width()<<endl;
			class_vectors.save_png(("pca_model." + c_iter->first + ".png").c_str());
	      }		
	}

	CImg<double> covariate(CImg<double> class_vectors)
	{
		CImg<double> multiplied(class_vectors.width(),class_vectors.width(),1)																																																																																																																																																																																																																																																																																																																																																											;
		for (int i=0; i<class_vectors.height(); i++) 
		{
			CImg<double> col_vector(class_vectors.width(),1,1); // cols = n, rows =1
			CImg<double> row_vector(1,class_vectors.width(),1); // cols = 1, rows = n
			for (int j=0; j<class_vectors.width(); j++)
			{																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																											
				row_vector(0,j) = class_vectors(j,i);
				col_vector(j,0) = class_vectors(j,i);
			}

			multiplied = multiplied + (row_vector * col_vector);
			//cout<<"rows and cols of matrix mux: "<<multiplied.width()<<"   "<<multiplied.height()<<endl;

		}
		return multiplied;
	}

	CImg<double> subtract_mean(CImg<double> class_vectors)
	{
		//initializing mean array to zero
		double meanArr[class_vectors.width()] ;
		for (int i=0;i< class_vectors.width();i++)
			meanArr[i] = 0;

		//for addition column wise
		for (int i=0; i<class_vectors.height(); i++) 
		{
			for (int j=0; j<class_vectors.width(); j++)
			{
				meanArr[j] = meanArr[j] + class_vectors(j,i);
			}
		}
		
		//calculating mean for each column
		int images = class_vectors.height();
		for (int i=0; i<class_vectors.width(); i++)
			meanArr[i] = meanArr[i]/images;

		//subtracting mean
		for (int i=0; i<class_vectors.height(); i++) 
		{
			for (int j=0; j<class_vectors.width(); j++)
			{
				class_vectors(j,i) = meanArr[j] - class_vectors(j,i);
			}
		}

		return class_vectors;
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

