#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


void linearFilter(Mat input, Mat kernel, Mat &output) {
    cout << "\nStarting linear filter operation" << endl;
    int input_n = input.rows;
    int input_m = input.cols;
    int kernel_n = kernel.rows;
    int kernel_m = kernel.cols;
    
    cout << "Input image size: " + to_string(input_n) + " x " + to_string(input_m) << endl;

    // Check if input image is grayscale
    if (input.type() != CV_8UC1) {
        cout << "Error: Input image not correct format!" << endl;
        return;
    }

    // Check if kernel is square
    if (kernel_n != kernel_m) {
        cout << "Error: Kernel matrix not square!" << endl;
        return;
    }

    // Check if kernel size is odd
    if ((kernel_n % 2 == 0)) {
        cout << "Error: Kernel size is even!" << endl;
        return;
    }

    cout << "Kernel size: " + to_string(kernel_n) + " x " + to_string(kernel_m) << endl;

    // Add padding to the input image
    int k = (kernel_n - 1) / 2;
    Mat padded_input = Mat::zeros(input_n+(2*k), input_m+(2*k), CV_8UC1);
    int padded_input_n = padded_input.rows;
    int padded_input_m = padded_input.cols;
    cout << "Padded input image size: " + to_string(padded_input_n) + " x " + to_string(padded_input_m) << endl;

    // Insert the input image into the center of the padded image
    input.copyTo(padded_input(Rect(k, k, input_m, input_n)));
    namedWindow("Padded input", WINDOW_NORMAL);
    imshow("Padded input", padded_input);
    waitKey(0);
 
    // Create output image
    output = Mat::zeros(input_n, input_m, CV_8UC1);
    for (int n=0; n<input_n; n++) {
        for (int m=0; m<input_m; m++) {

            // Apply the kernel to the pixel at (n,m)
            float sum = 0.0;
            
            for (int i=-k; i<=k; i++) {
                for (int j=-k; j<=k; j++) {
                    sum += padded_input.at<uchar>(n+k-i, m+k-j) * kernel.at<float>(i+k, j+k);
                }
            }
            //float weighted_sum = sum / (kernel_n * kernel_m); Removed due to Kernel already being normalized
            output.at<uchar>(n,m) = sum;
            
        }
    }
}


int main(int argc, char* argv[]) {
    //Load image as grayscale
    if (argc != 2) {
        cout << "Usage: ./main <imagefile.jpg/png>" << endl;
        return -1;
    }
    
    string filename = argv[1];
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    namedWindow("src");
    imshow("src", src);
    waitKey(0);
    CV_Assert(src.type() == CV_8UC1);

    //Create uniform 3x3 kernel
    Mat kernel(3,3, CV_32FC1, Scalar(1.0/9.0));
    CV_Assert(kernel.type() == CV_32FC1);

    //Apply linear filter
    Mat output;
    linearFilter(src, kernel, output);
    namedWindow("Linear filter output", WINDOW_NORMAL);
    imshow("Linear filter output", output);
    waitKey(0);

    //Test with custom kernel on impulse image
    Mat impulse = Mat::zeros(5,5,CV_8UC1);
    impulse.at<uchar>(2,2) = 1;

    Mat customKernel = (Mat_<float>(3,3) << 1,2,3,4,5,6,7,8,9);
    linearFilter(impulse, customKernel, output);

    for(int i=0; i<output.rows; i++){
        for(int j=0; j<output.cols; j++){
            cout << (int) output.at<uchar>(i,j) << " ";
        }
        cout << endl;
    }
    return 0;
}