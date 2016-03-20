#include <fstream>

class Classifier {
public:

    Classifier(const vector<string> &_class_list) : class_list(_class_list) {
    }

    // Run training on a given dataset.
    virtual void train(const Dataset &filenames) = 0;

    // Classify a single image.
    virtual string classify(const string &filename) = 0;

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
                predictions[c_iter->first][*f_iter] = classify(*f_iter);
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
    
    // testing new image using svm_multiclass_classify 
    void test_svm(const string test_image) {
        string tar_type = test_image.substr(0, 4);
        int target = convert(tar_type);
        if (target == 0) {
            cout << "Invalid test image input\nPlease rename the test image in proper format to check accuracy of detection..." << endl;
            return;
        }
        CImg<double> subsampled = CImg<double>(test_image.c_str()).resize(40, 40, 1, 3).unroll('x');
        std::ofstream ofs;
        ofs.open("test.dat");
        ofs << "#This is the first line...\n";
        ofs << target << " "; //target class is given 1 by default
        for (int i = 0; i < subsampled.size(); i++) {
            if (subsampled[i] != 0) {
                ofs << i + 1 << ":" << subsampled[i] << " ";
            }
        }
        ofs << "#abcde\n";
        ofs.close();
        if (check_file("test.dat") && check_file("model_file"))
            system("./svm_multiclass_classify test.dat model_file output_file");
        cout << "Output file generated..." << endl;
        if (check_file("output_file")) {
            ifstream inFile;
            inFile.open("output_file");
            int first;
            while (inFile >> first) {
                if (first == target)
                    cout << "Correctly classified...\nTest image belongs to class " << first << " : "<<tar_type<<endl;
                else
                    cout << "Class classified by svm : " <<first<< endl;
                inFile.ignore(numeric_limits<streamsize>::max(), '\n');
            }
        }
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
// to convert string class name to corresponding integer value
    int convert(const string name) {
        if (name == "bage")
            return 1;
        else if (name == "brea")
            return 2;
        else if (name == "brow")
            return 3;
        else if (name == "chic")
            return 4;
        else if (name == "chur")
            return 5;
        else if (name == "croi")
            return 6;
        else if (name == "fren")
            return 7;
        else if (name == "hamb")
            return 8;
        else if (name == "hotd")
            return 9;
        else if (name == "jamb")
            return 10;
        else if (name == "kung")
            return 11;
        else if (name == "lasa")
            return 12;
        else if (name == "muff")
            return 13;
        else if (name == "pael")
            return 14;
        else if (name == "pizz")
            return 15;
        else if (name == "popc")
            return 16;
        else if (name == "pudd")
            return 17;
        else if (name == "sala")
            return 18;
        else if (name == "salm")
            return 19;
        else if (name == "scon")
            return 20;
        else if (name == "spag")
            return 21;
        else if (name == "sush")
            return 22;
        else if (name == "taco")
            return 23;
        else if (name == "tira")
            return 24;
        else if (name == "waff")
            return 25;
        else
            return 0;
    }
protected:
    vector<string> class_list;
};
