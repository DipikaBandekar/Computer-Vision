
#ifndef SVM_H
#define SVM_H

#include "CImg.h"
#include <fstream>

class svm : public Classifier {
public:

    svm(const vector<string> &_class_list) : Classifier(_class_list) {
    }
    
// creating training.dat in format specified on website: https://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html
    
    virtual void train(const Dataset &filenames) {

        cout<<"empty train function"<<endl;

    }

    virtual void load_model() {
        cout << "empty load function" << endl;
    }

    virtual string classify(const string &filename, const Dataset &filenames) {
        cout << "empty classify function" << endl;
    }


protected:
    // extract features from an image, which in this case just involves resampling and 
    // rearranging into a vector of pixel data.

    CImg<double> extract_features(const string &filename) {
        return (CImg<double>(filename.c_str())).resize(size, size, 1, 3).unroll('x');
    }

    static const int size = 50; // subsampled image resolution
    map<string, CImg<double> > models; // trained models
};
#endif // SVM_H
