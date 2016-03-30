#include "CImg.h"
#include <Sift.h>

class bow : public Classifier {
public:

    bow(const vector<string> &_class_list) : Classifier(_class_list) {
    }

    //The train function performs kmean clustering on the training images by generating the sift descriptors 
    //and then generating the means and adjusting the vector count according to clustered indexes.

    virtual void train(const Dataset &filenames) {
        std::ofstream ofs;
        int target = 1;
        ofs.open("training_bow.dat");
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
            // cout<<"inside final loop";
            std::vector<int> ctoff(25, 0);
            int maxtrgt = 0;
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
        if (ofs.eof())
            cout << "Training file is empty..." << endl;
        else //executing svm_multiclass_learn to generate model file
            system("./svm_multiclass_learn -c 1.0 training_bow.dat model_bow_file");
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
            //  cout<<"Assigning clusters"<<endl;
            // cout << "Processing " << c_iter->first << endl;
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

