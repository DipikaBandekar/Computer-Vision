#include <fstream>
#include <vector>
#include "CImg.h"
#include <stdio.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <iterator>
#include <string>
#include <stdlib.h>

class Classifier {
public:

    Classifier(const vector<string> &_class_list) : class_list(_class_list) {
    }

    // Run training on a given dataset.
    virtual void train(const Dataset &filenames) = 0;

    // Classify a single image.
    virtual string classify(const string &filename, const Dataset &filenames) = 0;

    //virtual void testEigen(const Dataset &filenames) =0;

    // Load in a trained model.
    virtual void load_model() = 0;

    // Loop through all test images, hiding correct labels and checking if we get them right.

    void test(const Dataset &filenames) {
        cerr << "Loading model..." << endl;
        load_model();

        // loop through images, doing classification
        map<string, map<string, string> > predictions;
        for (map<string, vector<string> >::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
            for (vector<string>::const_iterator f_iter = c_iter->second.begin(); f_iter != c_iter->second.end(); ++f_iter) {
                cerr << "Classifying " << *f_iter << "..." << endl;
                predictions[c_iter->first][*f_iter] = classify(*f_iter, filenames);
            }

        // now score!
        map< string, map< string, double > > confusion;
        int correct = 0, total = 0;
        for (map<string, vector<string> >::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
            for (vector<string>::const_iterator f_iter = c_iter->second.begin(); f_iter != c_iter->second.end(); ++f_iter, ++total)
                confusion[c_iter->first][ predictions[c_iter->first][*f_iter] ]++;

        cout << "Confusion matrix:" << endl << setw(20) << " " << " ";
        for (int j = 0; j < class_list.size(); j++)
            cout << setw(2) << class_list[j].substr(0, 2) << " ";

        for (int i = 0; i < class_list.size(); i++) {
            cout << endl << setw(20) << class_list[i] << " ";
            for (int j = 0; j < class_list.size(); j++)
                cout << setw(2) << confusion[ class_list[i] ][ class_list[j] ] << (j == i ? "." : " ");

            correct += confusion[ class_list[i] ][ class_list[i] ];
        }

        cout << endl << "Classifier accuracy: " << correct << " of " << total << " = " << setw(5) << setprecision(2) << correct / double(total)*100 << "%";
        cout << "  (versus random guessing accuracy of " << setw(5) << setprecision(2) << 1.0 / class_list.size()*100 << "%)" << endl;
    }

    virtual string train_test_svm(CImg<double> inputImg, map<int, int> imageClass_map, string value, const Dataset &filenames, const string &filename)//, const string value) {
    {
        int target = 1;
        vector <int> target_arr;
        int count = 0;
        std::ofstream ofs;
        if (value == "train")
            ofs.open("train.dat");
        if (value == "test")
            ofs.open("test.dat");
        ofs << "#This is the first line...\n";
        cout << "creating a new train file" << endl;
        int classNo = 0;
        if (value == "test") {
            int flag = 0;

            for (map<string, vector<string> >::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter) {
                classNo++;
                for (vector<string>::const_iterator f_iter = c_iter->second.begin(); f_iter != c_iter->second.end(); ++f_iter) {
                    //cerr << "Classifying " << *f_iter << "..." << endl;
                    // cout<<f_iter->c_str()<<endl;
                    // cout<<"comparison result: "<<std::strcmp(f_iter->c_str(),filename.c_str())<<endl;
                    if ((std::strcmp(f_iter->c_str(), filename.c_str())) == 0) {
                        cout << "class number found: " << classNo << endl;
                        flag = 1;
                        break;
                    }
                }
                if (flag == 1)
                    break;
            }
            cout << "classNo found was: " << classNo << endl;
            for (int j = 0; j < inputImg.height(); j++) {
                ofs << classNo << " ";

                for (int k = 0; k < inputImg.width(); k++) {
                    ofs << k + 1 << ":" << inputImg(k, j) << " ";
                }
                ofs << "#" << j << "\n";
            }
            target_arr.push_back(target);

        } else if (value == "train") {
            for (int j = 0; j < inputImg.height(); j++) {
                //target_arr.push_back(target);
                //count++;
                ofs << imageClass_map.find(j)->second << " ";
                for (int k = 0; k < inputImg.width(); k++) {
                    ofs << k + 1 << ":" << inputImg(k, j) << " ";
                }
                ofs << "#" << j << "\n";
            }
        }
        //target = target + 1;
        ofs.close();

        int c = 0;
        int correct = 0;
        if (value == "test") {
            if (check_file("test.dat") && check_file("model_file"))
                system("./svm_multiclass_classify test.dat model_file prediction_file");
            cout << "prediction file generated..." << endl;
            // checking the accuracy of detection...
            if (check_file("prediction_file") && check_file("test.dat")) {
                ifstream out_file;
                out_file.open("prediction_file");
                int first;
                while (out_file >> first && c < target_arr.size()) {
                    if (static_cast<int> (first) == target_arr[c]) {
                        correct++;
                    }
                    c++;
                    out_file.ignore(numeric_limits<streamsize>::max(), '\n');
                }
                out_file.close();
                float accuracy = 0.0;
                if (c == target_arr.size())
                    accuracy = (static_cast<float> (correct) / static_cast<float> (c)) * 100.0;
                cout << "Accuracy of SVM Classifier:: " << accuracy << "%" << endl;
            }


            ifstream outputPred("prediction_file");
            int val;
            outputPred >>val;
            return class_list[val - 1];
        } else if (value == "train") {
            system("./svm_multiclass_learn -c 1.0 train.dat model_file");

        } else
            cout << "Error" << endl;
        return "";
    }
    //for all test images

    void svm(const Dataset & filenames, string value) {
        int target = 1;
        vector <int> target_arr;
        int count = 0;
        std::ofstream ofs;
        if (value == "train")
            ofs.open("train_svm.dat");
        else
            ofs.open("test_svm.dat");
        ofs << "#This is the first line...\n";
        for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter) {

            cout << "Processing svm model " << c_iter->first << " : " << target << endl;

            for (int i = 0; i < c_iter->second.size(); i++) {
                CImg<double> temp1 = (CImg<double>(c_iter->second[i].c_str())).resize(50, 50, 1, 3).unroll('x');
                target_arr.push_back(target);
                count++;
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

        int c = 0;
        int correct = 0;
        if (value == "test") {
            if (check_file("test_svm.dat") && check_file("model_file_svm"))
                system("./svm_multiclass_classify test_svm.dat model_file_svm prediction_file");

            cout << "prediction file generated..." << endl;
            // checking the accuracy of detection...
            if (check_file("prediction_file") && check_file("test_svm.dat")) {
                ifstream out_file;
                out_file.open("prediction_file");
                int first;
                while (out_file >> first && c < target_arr.size()) {
                    if (static_cast<int> (first) == target_arr[c]) {
                        correct++;
                    }
                    c++;
                    out_file.ignore(numeric_limits<streamsize>::max(), '\n');
                }
                out_file.close();
                float accuracy = 0.0;
                if (c == target_arr.size())
                    accuracy = (static_cast<float> (correct) / static_cast<float> (c)) * 100.0;
                cout << "Accuracy of SVM Classifier:: " << accuracy << "%" << endl;
            }
        } else if (value == "train") {
            system("./svm_multiclass_learn -c 1.0 train_svm.dat model_file_svm");

        } else
            cout << "Error" << endl;
    }
    //check whether a file exists

    bool check_file(string input_file) {
        ifstream f(input_file.c_str());
        if (f.good()) {
            f.close();
            return true;
        } else {
            cout << input_file << " does not exist!!!" << endl;
            f.close();
            return false;
        }
    }

    void train_test_deep(const Dataset &filenames, string value) {
        int target = 1;
        vector <int> target_arr;
        int count = 0;
        std::ofstream ofs;
        if (value == "train")
            ofs.open("train_deep.dat");
        else
            ofs.open("test_deep.dat");
        ofs << "#This is the first line...\n";
        for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter) {

            cout << "Processing deep network model " << c_iter->first << " : " << target << endl;

            for (int i = 0; i < c_iter->second.size(); i++) {
                CImg<double> resize_image = (CImg<double>(c_iter->second[i].c_str())).resize(250, 250, 1, 3);
                target_arr.push_back(target);
                resize_image.save_png(("deep.png"));
                //cout << "resized image saved..." << endl;
                system(" ./overfeat/bin/linux_64/overfeat -f deep.png > deep_features");
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
                int feature = 1;
                // Read an integer at a time from the line
                for (int i = 0; i < tokens.size(); i++) {
                    if (tokens[i] != "0") {
                        //std::cout << feature << ":" << tokens[i] << " ";
                        ofs << feature << ":" << tokens[i] << " ";
                        feature++;
                    }
                }
                ofs << "#" << c_iter->first << "\n";
            }
            target = target + 1;
        }
        ofs.close();
        int c = 0;
        int correct = 0;
        if (value == "train")
            system("./svm_multiclass_learn -c 1.0 train_deep.dat model_file_deep");
        else if (value == "test") {
            if (check_file("test_deep.dat") && check_file("model_file_deep"))
                system("./svm_multiclass_classify test_deep.dat model_file_deep prediction_file_deep");

            cout << "prediction file generated..." << endl;
            // checking the accuracy of detection...
            if (check_file("prediction_file_deep") && check_file("test_deep.dat")) {
                ifstream out_file;
                out_file.open("prediction_file_deep");
                int first;
                while (out_file >> first && c < target_arr.size()) {
                    if (static_cast<int> (first) == target_arr[c]) {
                        correct++;
                    }
                    c++;
                    out_file.ignore(numeric_limits<streamsize>::max(), '\n');
                }
                out_file.close();
                float accuracy = 0.0;
                if (c == target_arr.size())
                    accuracy = (static_cast<float> (correct) / static_cast<float> (c)) * 100.0;
                cout << "Accuracy of Deep Classifier:: " << accuracy << "%" << endl;
            }
        }

    }

    void train_test_haar(const Dataset &filenames, string value) {
        int target = 1;
        vector <int> target_arr;
        int count = 0;
        std::ofstream ofs;
        if (value == "train")
            ofs.open("train_haar.dat");
        else
            ofs.open("test_haar.dat");
        ofs << "#This is the first line...\n";
        for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter) {

            cout << "Processing haar network model " << c_iter->first << " : " << target << endl;

            for (int i = 0; i < c_iter->second.size(); i++) {
                CImg<double> resize_image = (CImg<double>(c_iter->second[i].c_str())).resize(50, 50, 1, 3);
                target_arr.push_back(target);
                // calculating differences
                vector<double> features;
                for (int m = 0; m < 1000; m++) {
                    int max_ht = resize_image.height();
                    int max_wd = resize_image.width();
                    int x = rand() % 35;
                    int y = rand() % 35;
                    int height = rand() % 8;
                    int width = rand() % 6;
                    double total_pixels_1 = 0.0, total_pixels_2 = 0.0;
                    if ((x + width) < max_wd && (y + (2 * height)) < max_ht) {
                        for (int j = 0; j < width; j++) {
                            for (int k = 0; k < height; k++) {
                                total_pixels_1 = total_pixels_1 + resize_image((x + j), (y + k));
                                total_pixels_2 = total_pixels_2 + resize_image((x + j), (y + height + k));
                            }
                        }
                    }
                    double difference = abs(total_pixels_2 - total_pixels_1);
                    //cout << i << " :: " << difference << endl;
                    features.push_back(difference);
                }
                //  cout << i << " : " << features.size() << endl;
                ofs << target << " ";
                for (int n = 0; n < features.size(); n++) {
                    if (features[n] != 0)
                        ofs << n + 1 << ":" << features[n] << " ";
                }
                ofs << "#" << c_iter->first << "\n";
            }
            target = target + 1;
        }
        ofs.close();
        int c = 0;
        int correct = 0;
        if (value == "train")
            system("./svm_multiclass_learn -c 1.0 train_haar.dat model_file_haar");
        else if (value == "test") {
            if (check_file("test_haar.dat") && check_file("model_file_haar"))
                system("./svm_multiclass_classify test_haar.dat model_file_haar prediction_file_haar");

            cout << "prediction file generated..." << endl;
            // checking the accuracy of detection...
            if (check_file("prediction_file_haar") && check_file("test_haar.dat")) {
                ifstream out_file;
                out_file.open("prediction_file_haar");
                int first;
                while (out_file >> first && c < target_arr.size()) {
                    if (static_cast<int> (first) == target_arr[c]) {
                        correct++;
                    }
                    c++;
                    out_file.ignore(numeric_limits<streamsize>::max(), '\n');
                }
                out_file.close();
                float accuracy = 0.0;
                if (c == target_arr.size())
                    accuracy = (static_cast<float> (correct) / static_cast<float> (c)) * 100.0;
                cout << "Accuracy of Haar Classifier:: " << accuracy << "%" << endl;
            }

        }
    }

    void test_bow(const Dataset &filenames) {
        std::ofstream ofs;
        int target = 1;
        vector <int> target_arr;
        ofs.open("test_bow.dat");
        ofs << "#This is the first line...\n";
        vector<SiftDescriptor> masterdesc;
        vector<float> mastermeans;
        //to get the count of all the descriptors in all the images
        int count = 0;
        int imageno = 0;
        vector<pair<float, float> > descriptorno;
        //iterate through all the files to get the images
        for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter) {
            cout << "Processing " << c_iter->first << endl;
            // convert each image to be a row of this "model" image
            for (int i = 0; i < c_iter->second.size(); i++) {
                CImg<double> gray(c_iter->second[i].c_str());
                CImg<double> temp1 = gray.get_RGBtoHSI().get_channel(2);
                //generating the sift descriptors for the image
                masterdesc = Sift::compute_sift(temp1);
                //generate the mean for all the descriptors
                for (int k = 0; k < masterdesc.size(); k++) {
                    float sum = 0;
                    for (int l = 0; l < 128; l++) {
                        //   cout<<l<<":"<<masterdesc[k].descriptor[l]<<" ; ";
                        sum = sum + masterdesc[k].descriptor[l];
                    }
                    sum = (sum / 128);
                    //  cout<<"sum"<<sum<<endl;
                    mastermeans.push_back(sum);
                    //    cout<<mastermeans[count]<<endl;
                    count++;
                }
                //creating a pair for every descriptor number and image
                pair<int, int> match;
                match.first = imageno;
                match.second = masterdesc.size();
                descriptorno.push_back(match);
                imageno++;
                // cout<<imageno;
            }
        }
        count = count - 1;
        cout << "created the mean matrix for images of count" << count << endl;
        //creating k clusters of centroids and randomly getting 25 of them
        vector<float> means = kmean_process(mastermeans);
        cout << "mean size" << means.size() << endl;
        //passing the old values of means and then clustering group of descriptors to each centroid
        vector< vector<int> > assignedclusters = assign_clusters(means, filenames, imageno);
        cout << "calculated clusters" << endl;
        vector<float > newmeans;
        int i = 0;
        //rebealancing the mean for the clusters and reassigning the descriptors to each cluster
        while (i < 2) {
            newmeans = calc_newcentroids(means, assignedclusters, mastermeans);
            assignedclusters = assign_clusters(newmeans, filenames, imageno);
            i++;
        }
        int mainclass = 0;
        //iterate through every image
        for (int imgg = 0; imgg < imageno; imgg++) {
            //keep the maximum class label
            //  cout<<"inside final loop";
            int maxtrgt = 0;
            std::vector<int> ctoff(25, 0);
            for (int ct = 0; ct < 25; ct++) {
                //   for(int i=0;i<assignedclusters.size();i++){
                for (int j = 0; j < assignedclusters[imgg].size(); j++) {
                    if (assignedclusters[imgg][j] == ct)
                        ctoff[ct] = ctoff[ct] + 1;
                    else if (assignedclusters[imgg][j] == -1)
                        break;
                }
                if (ctoff[ct] > maxtrgt) {
                    maxtrgt = ctoff[ct];
                    mainclass = ct + 1;
                }
            }
            ofs << mainclass << " ";
            for (int ot = 0; ot < 25; ot++) {
                ofs << ot + 1 << ":" << ctoff[ot] << " ";
            }
            ofs << "#" << "one" << "\n";
        }
        ofs.close();
        float c = 0;
        float correct = 0;
        if (ofs.eof())
            cout << "test file is empty..." << endl;
        else //executing svm_multiclass_learn to generate model file
            if (check_file("test_bow.dat") && check_file("model_bow_file"))
            system("./svm_multiclass_classify test_bow.dat model_bow_file prediction_file_bow");

        cout << "prediction file generated..." << endl;
        // checking the accuracy of detection...
        if (check_file("prediction_file_bow") && check_file("test_bow.dat")) {
            ifstream out_file;
            out_file.open("prediction_file_bow");
            int first;
            while (out_file >> first && c < target_arr.size()) {
                if (static_cast<int> (first) == target_arr[c]) {
                    correct++;
                }
                c++;
                out_file.ignore(numeric_limits<streamsize>::max(), '\n');
            }
            out_file.close();
            float accuracy = 0.0;
            if (c == target_arr.size())
                accuracy = (static_cast<float> (correct) / static_cast<float> (c)) * 100.0;
            cout << "Accuracy of Bag of Words Classifier:: " << accuracy << "%" << endl;
        }

    }



    //create the k clusters with centroids

    vector<float> kmean_process(vector<float> &mastermeans) {
        vector<float> centroids;
        for (int i = 0; i < 25; i++) {
            int randomindex = rand() % mastermeans.size();
            centroids.push_back(mastermeans[randomindex]);
        }
        return centroids;
    }


    //choosing the centroid cluster which is going to be the minimum distance for every descriptor

    vector< vector<int> > assign_clusters(vector<float> means, const Dataset &filenames, int imagect) {
        vector<float> oldmeans = means;
        // vector< vector<int> > centroidval;
        //std::vector<std::vector<float>> centroidval(imagect,0);
        std::vector< std::vector<int> > centroidval(imagect, std::vector<int>(20000, -1));
        vector<SiftDescriptor> masterdesc;
        int imageno = 0;
        for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter) {
            //     cout<<"Assigning clusters"<<endl;
            //   cout << "Processing " << c_iter->first << endl;
            // convert each image to be a row of this "model" image
            for (int i = 0; i < c_iter->second.size(); i++) {
                CImg<double> gray(c_iter->second[i].c_str());
                CImg<double> temp1 = gray.get_RGBtoHSI().get_channel(2);
                //pushing in the Sift descriptors
                masterdesc = Sift::compute_sift(temp1);
                //                cout<<"size of descriptors for image "<<c_iter->second[i].c_str() <<" : "<<masterdesc.size()<<endl;
                for (int descr = 0; descr < masterdesc.size(); descr++) {
                    float sum = 0;
                    float minimumcluster = 999999;
                    int index = 5;
                    for (int k = 0; k < means.size(); k++) {
                        for (int l = 0; l < 128; l++) {
                            float diff = (masterdesc[descr].descriptor[l]) - means[k];
                            sum = sum + abs(diff);
                        }
                        float val = sqrt(sum);
                        // cout<<"sum:"<<val<<endl;
                        if (val < minimumcluster) {
                            //  cout<<"haha";
                            minimumcluster = val;
                            index = k;
                        }
                    }
                    //cout<<"getting value for mean for image " <<imageno<<" descriptor no "<<descr<<" : "<<index<<endl;
                    centroidval[imageno][descr] = index;
                }
                imageno++;
            }
        }

        return centroidval;
    }


    //recentering the clusters to balanced centroid representative  of all descriptors

    vector<float> calc_newcentroids(vector<float> means, vector< vector<int> > assignedclusters, vector<float> mastermeans) {
        // cout<<"entered";
        vector<float> oldmeans(means);
        vector<pair<int, int> > indexassigned;
        // cout<<"assigned oldmeans";
        for (int i = 0; i < 25; i++) {
            float count = 0;
            float meanie = 0;
            for (int imgno = 0; imgno < assignedclusters.size(); imgno++) {
                for (int descr = 0; descr < assignedclusters[imgno].size(); descr++) {
                    if (assignedclusters[imgno][descr] == i) {
                        count = count + 1;
                        meanie = meanie + mastermeans[(assignedclusters.size() * imgno) + descr];
                    }
                    else if (assignedclusters[imgno][descr] == -1)
                        break;
                }
            }
            means[i] = meanie / count;

        }
        return means;
    }


protected:
    vector<string> class_list;
};
