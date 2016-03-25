
class pca : public Classifier {
   public:
	pca(const vector<string> &_class_list) : Classifier(_class_list)
	{}
	
	virtual void train(const Dataset &filenames) 
	{
         for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
	     {
			cout << "Processing " << c_iter->first << endl;
			CImg<double> class_vectors(size*size, c_iter->second.size(), 1);

			// convert each image to be a row of this "model" image
			for(int i=0; i<c_iter->second.size(); i++)
			  class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));
			
			class_vectors = subtract_mean(class_vectors); //50 X 1600
			CImg<double> mul = covariate(class_vectors);  // 1600 X 1600
			
			//getting the eigen values
			CImg<double> eigen(class_vectors.width(),class_vectors.width(),1);
			mul.symmetric_eigen(eigen,eigen);

			int k = size*size/4;
			cout<<"k: "<<k<<endl;

			//reducing the eigen matrix from n X n to n X k
			CImg<double> reducedEigen(k,eigen.height(),1);
			for(int i=0; i<eigen.height();i++)
			{
				for(int j=0; j<k; j++)
					reducedEigen(j,i) = eigen(j,i);
			}

			CImg<double> pcaMatrix = class_vectors * reducedEigen;

			cout<<"size of reduced Eigen: "<<pcaMatrix.height()<<" X "<<pcaMatrix.width()<<endl;

			//class_vectors.save_png(("pca_model." + c_iter->first + ".png").c_str());
	      }		
	}

	CImg<double> covariate(CImg<double> class_vectors)
	{
		CImg<double> multiplied(class_vectors.width(),class_vectors.width(),1);																																																																																																																																																																																																																																																																																																																																																											;
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

	CImg<double> subtract_mean(CImg<double> c_vectors)
	{
		//initializing mean array to zero
		double meanArr[c_vectors.width()] ;
		for (int i=0;i< c_vectors.width();i++)
			meanArr[i] = 0;

		//for addition column wise
		for (int i=0; i<c_vectors.height(); i++) 
		{
			for (int j=0; j<c_vectors.width(); j++)
			{
				meanArr[j] = meanArr[j] + c_vectors(j,i);
			}
		}
		
		//calculating mean for each column
		int images = c_vectors.height();
		for (int i=0; i<c_vectors.width(); i++)
			meanArr[i] = meanArr[i]/images;

		//subtracting mean
		for (int i=0; i<c_vectors.height(); i++) 
		{
			for (int j=0; j<c_vectors.width(); j++)
			{
				c_vectors(j,i) = meanArr[j] - c_vectors(j,i);
			}
		}

		return c_vectors;
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

