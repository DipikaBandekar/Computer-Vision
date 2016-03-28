
/* 
 * File:   deep.h
 * Author: dipika
 *
 * Created on March 26, 2016, 12:26 AM
 */

#ifndef DEEP_H
#define DEEP_H

#include "CImg.h"
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <limits>
#include <vector>
#include <sstream>
#include <iterator>
#include <string>

class deep : public Classifier {
public:

    deep(const vector<string> &_class_list) : Classifier(_class_list) {
    }

    // creating training.dat in format specified on website: https://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html

    virtual void train(const Dataset &filenames) {
        int target = 1;
        std::ofstream ofs;
        ofs.open("train_deep.dat");
        ofs << "#This is the first line...\n";
        for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter) {

            cout << "Processing deep network model " << c_iter->first << " : " << target << endl;

            for (int i = 0; i < c_iter->second.size(); i++) {
                CImg<double> resize_image = (CImg<double>(c_iter->second[i].c_str())).resize(size, size, 1, 3);
                resize_image.save_png(("deep.png"));
                //cout << "resized image saved..." << endl;
                system(" ./overfeat/bin/linux_64/overfeat -L 12 deep.png > deep_features");
                //cout << "executed overfeat and copied to deep features" << endl;
                ifstream out_file;
                out_file.open("deep_features");
                out_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                string line;
                getline(out_file, line);
                std::istringstream buf(line);
                std::istream_iterator<std::string> beg(buf), end;
                std::vector<std::string> tokens(beg, end); // done
                ofs << target << " ";
                int value;
                int feature = 1;
                // Read an integer at a time from the line
                for (int i = 0; i < tokens.size(); i++) {
                    if(tokens[i]!="0"){
                    std::cout << feature << ":" << tokens[i] << " ";
                    ofs << feature << ":" << tokens[i] << " ";
                    feature++;
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
        system("./svm_multiclass_learn -c 1.0 train_deep.dat model_file_deep");
}

virtual void load_model() {
    cout << "empty load function" << endl;
}

virtual string classify(const string &filename) {
    cout << "empty classify function" << endl;
}


protected:
//extract features from an image, which in this case just involves resampling and 
// rearranging into a vector of pixel data.

static const int size = 231; // subsampled image resolution
map<string, CImg<double> > models; // trained models
};

#endif // DEEP_H