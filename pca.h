
#include <string>
#include <sstream>
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
			
			CImg<double> mean_subtracted_vector = subtract_mean(class_vectors); //50 X 1600
			CImg<double> mul = covariate(mean_subtracted_vector);  // 1600 X 1600
			
			CImg<double> eigen_vec;
			CImg<double> eigen_val;
			mul.symmetric_eigen(eigen_val,eigen_vec);
			//cout<<"size of eigen_val: "<<eigen_val.height()<<" X "<<eigen_val.width()<<endl;
			cout<<"size of eigen_vec: "<<eigen_vec.height()<<" X "<<eigen_vec.width()<<endl;
			int k = size*size/4;
			cout<<"k: "<<k<<endl;

			//reducing the eigen matrix from n X n to n X k
			CImg<double> reducedEigen(k,eigen_vec.height(),1,1,0); //1600 X k
			for(int i=0; i<eigen_vec.height();i++)
			{
				for(int j=0; j<k; j++)
					reducedEigen(j,i) = eigen_vec(j,i);
			}

			CImg<double> pcaMatrix = mean_subtracted_vector * reducedEigen; //  
			
			cout<<"Saving the pcaMatrix"<<endl;
			pcaMatrix.normalize(0,255);
			cout<<"printing pcaMatrix: "<<endl;
			printMatrix(pcaMatrix);
			pcaMatrix.save_png("savedPca.png");
			
			CImg<double> transposed = pcaMatrix.transpose();
			CImgList<double> c_list = roll(transposed);
			//for(CImgList::const_iterator it1 =c_list.begin(); it1 != c_list.end(); ++it1)
			printImageFromList(c_list);
			
			break;
			
	      }		
	}

	void printImageFromList(CImgList<double> c_list)
	{
		for(int m=0; m<c_list.size(); m++)
			{
				std::stringstream ss;
				ss << m;
				std::string st;
				ss>>st;
				c_list[m].save_png(("rolled" + st + ".png").c_str());

			}
	}

	void printMatrix(CImg<double> pcaMatrix)
	{
		for (int i =0; i< pcaMatrix.height(); i++)
			{
				cout<<endl;
				for( int j=0; j< pcaMatrix.width(); j++)
				{
					cout<<pcaMatrix(j,i)<<"   ";
				}
			}
	}

	CImgList<double> roll(CImg<double> image) 
	{
	    CImgList<double> list;
        for (int x = 0; x < image.width(); x++) {
            CImg<double> new_image(size, size, 1, 1,0);
            int j = 0;
            int k = 0;
            for (; j < image.height();) {
                for (int i = 0; i < size; i++) {
                    new_image(i, k, 0, 0) = image(x, j, 0, 0);
                    j++;
                }
                k++;
            }
            list.push_back(new_image);
        }
        cout<<"sizeofList: "<<list.size()<<endl;
        return list;
    }


	CImg<double> covariate(CImg<double> class_vectors)
	{
		CImg<double> multiplied(class_vectors.width(),class_vectors.width(),1,1,0);																																																																																																																																																																																																																																																																																																																																																											;
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

		int imageCount = class_vectors.height();
		for (int i=0; i<multiplied.width(); i++)
		{
			for(int j=0; j<multiplied.height(); j++)
			{
				multiplied(j,i) = multiplied(j,i)/imageCount;
			}
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
				c_vectors(j,i) = c_vectors(j,i) - meanArr[j] ;
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

