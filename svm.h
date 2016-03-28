
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
        int target = 1;
        std::ofstream ofs;
        ofs.open("training.dat");
        ofs << "#This is the first line...\n";
        for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter) {
            cout << "Processing svm model " << c_iter->first << " : " << target << endl;

            for (int i = 0; i < c_iter->second.size(); i++) {
                CImg<double> temp1 = extract_features(c_iter->second[i].c_str());
                //cout << "Size " << temp1.size() << endl;
                ofs << target << " ";
                for (int j = 0; j < temp1.size(); j++) {
                    if (temp1[j] != 0) {
                        ofs << j + 1 << ":" << temp1[j] << " ";
                    }
                }
                ofs << "#" << c_iter->first << "\n";
            }
            target = target + 1;
        }
        ofs.close();
        if (ofs.eof())
            cout << "Training file is empty..." << endl;
        else //executing svm_multiclass_learn to generate model file
            system("./svm_multiclass_learn -c 1.0 training.dat model_file");
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

    static const int size = 40; // subsampled image resolution
    map<string, CImg<double> > models; // trained models
};
#endif // SVM_H
