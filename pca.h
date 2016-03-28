
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
class pca : public Classifier {
	CImgList<double> trainList;
	CImgList<double> testList;
   public:
	pca(const vector<string> &_class_list) : Classifier(_class_list)
	{}
	CImg<double> reducedEigenMatrix;
	
	virtual void train(const Dataset &filenames) 
	{
		cout<<"Inside the training of pca"<<endl;
		int z=0;
		int classNo = 1;
		int imgCount =0;
		cout<<"filenames.size(): "<<filenames.size()<<endl;
		for (map<string, vector<string> >::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
            for (vector<string>::const_iterator f_iter = c_iter->second.begin(); f_iter != c_iter->second.end(); ++f_iter)
            		imgCount++;
		CImg<double> class_vectors(size*size,imgCount,1);
		 for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
	     {
			cout << "Processing " << c_iter->first << endl;
			//CImg<double> class_vectors(size*size, c_iter->second.size(), 1);

			// convert each image to be a row of this "model" image
			for(int i=0; i<c_iter->second.size(); i++)
			{
				class_vectors = class_vectors.draw_image(0, z, 0, 0, extract_features(c_iter->second[i].c_str()));	
				imageClass_map.insert(std::pair<int,int>(z,classNo));
				z++;
			}
			classNo++;
			  
		}

			cout<<"size of class_vectors: "<<class_vectors.height()<<" X "<<class_vectors.width()<<endl;
			CImg<double> mean_subtracted_vector = subtract_mean(class_vectors); //1250 X 1600
			CImg<double> mul = covariate(mean_subtracted_vector);  // 1600 X 1600
			
			CImg<double> eigen_vec;
			CImg<double> eigen_val;
			mul.symmetric_eigen(eigen_val,eigen_vec);
			//cout<<"size of eigen_val: "<<eigen_val.height()<<" X "<<eigen_val.width()<<endl;
			cout<<"size of eigen_vec: "<<eigen_vec.height()<<" X "<<eigen_vec.width()<<endl;
			//int k = size*size/4;
			cout<<"k: "<<k<<endl;

			//reducing the eigen matrix from n X n to n X k
			CImg<double> reducedEigen(k,eigen_vec.height(),1,1,0); //1600 X k
			for(int i=0; i<eigen_vec.height();i++)
			{
				for(int j=0; j<k; j++)
					reducedEigen(j,i) = eigen_vec(j,i);
			}
			reducedEigenMatrix = reducedEigen; 
			cout<<"size of reducedEigenmatrix: "<<reducedEigenMatrix.height()<<" X "<<reducedEigenMatrix.width()<<endl;
			
			//saving image 
			cout<<"Saving reducedEigenmatrix"<<endl;
			writeToFile();
			//reducedEigenMatrix.save_png("reducedEigenMatrix.txt");
			CImg<double> pcaMatrix = class_vectors * reducedEigen; //  
	      
	      	train_test_svm(pcaMatrix,imageClass_map,"train",filenames,"");
	}
	void writeToFile()
	{
		ofstream f;
		f.open("reducedEigenmatrix.txt");
		for(int i=0;i<reducedEigenMatrix.height();i++)
		{
			for(int j=0; j<reducedEigenMatrix.width();j++)
			{
				f<<reducedEigenMatrix(j,i)<<" ";
			}
			f << '\n';
		}
		f.close();
	}


	CImg<double> readFromFile()
	{
			vector<string> tokens;
			vector<string> mainVector;
			string line;
			CImgList<double> eigenVector; 
			//char delim  = " ";
			ifstream myFile("reducedEigenmatrix.txt");
			if(myFile.is_open())
			{
				while(getline(myFile,line))
				{
					CImg<double> temp(k,1);
						tokens = split(line,' ');
						//cout<<"size of tokens: "<<tokens.size()<<endl;
						for(int i=0; i<tokens.size();i++)
							temp(i,0) = atof(tokens[i].c_str());
						eigenVector.push_back(temp);
				}
			}
			cout<<"size of mainVector :"<<mainVector.size()<<endl;
			CImg<double> eigen_vector = eigenVector.get_append('y');
			return eigen_vector;
	}

	std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
	{
	    std::stringstream ss(s);
	    std::string item;
	    while (std::getline(ss, item, delim)) {
	        elems.push_back(item);
	    }
	    return elems;
	}


	std::vector<std::string> split(const std::string &s, char delim) 
	{
	    std::vector<std::string> elems;
	    split(s, delim, elems);
	    return elems;
	}

	CImg<double> convertToGreyScale(const string &filename){
        
        CImg<double> image(filename.c_str());
        CImg<double> grayScaleImg;
        grayScaleImg=image.get_RGBtoYCbCr().get_channel(0);
        //grayScaleImg.save_png("grayscale.png");
        return grayScaleImg;//.resize(size, size, 1, 1).unroll('x');
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
        cout << "loading the reduced Eigen matrix" << endl;
        // const CImg<double> temp("reducedEigenMatrix.png");
        // cout<<"temp width and height"<<temp.width()<<"  "<<temp.height()<<endl;
        // models["eigenMatrix"] = temp;

    }

    virtual string classify(const string &filename,const Dataset &filenames) 
    {
    	CImg<double> currImg = extract_features(filename);
    	//const CImg<double> tmp("reducedEigenMatrix.txt");// = models.find("eigenMatrix")->second;
	
		CImg<double> tmp = readFromFile();
		CImg<double> newImg = currImg * tmp;// will give you 1 X k
    	cout<<"newImg.height() and width()"<<newImg.height()<<" X "<<newImg.width()<<endl;
    	int classNo=0;
    	cout<<"filename is : "<<filename<<endl;


        
        return train_test_svm(newImg,imageClass_map,"test",filenames,filename);
    }

   protected:
	CImg<double> extract_features(const string &filename) 
	{
		CImg<double> grey = convertToGreyScale(filename);
        	return grey.resize(size, size, 1, 1).unroll('x');
    	}
    static const int k = 400;
    map<int,int> imageClass_map;
    static const int size = 40; // subsampled image resolution
    map<string, CImg<double> > models; // trained models

};

