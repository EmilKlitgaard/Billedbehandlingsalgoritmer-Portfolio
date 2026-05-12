#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


// ==================================================== //
// ================= HELPER FUNCTIONS ================= //
// ==================================================== //

// Remove noice from image using morphological operations
static cv::Mat removeNoice(const cv::Mat &src) {
    cv::Mat cleanedBoard;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(src, cleanedBoard, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(cleanedBoard, cleanedBoard, cv::MORPH_CLOSE, kernel);
    return cleanedBoard;
}

// Convert the input image to a binary image highlighting the Sudoku grid
static cv::Mat convertToBinary(const cv::Mat &src) {
    // Convert to grayscale
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        cout << "Warning: Input image is not 3-channel, skipping grayscale conversion." << endl;
        gray = src.clone();
    }

    // Segment the board using 
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 31, 7);
    
    return binary;
}

// Order the corners of the detected rectangle
static vector<cv::Point2f> orderCorners(const vector<cv::Point2f> &corners) {
    vector<cv::Point2f> ordered(4);
    vector<float> sums, diffs;
    for (const auto &corner : corners) {
        sums.push_back(corner.x + corner.y);
        diffs.push_back(corner.y - corner.x);
    }

    ordered[0] = corners[min_element(sums.begin(), sums.end()) - sums.begin()]; // top-left
    ordered[2] = corners[max_element(sums.begin(), sums.end()) - sums.begin()]; // bottom-right
    ordered[1] = corners[min_element(diffs.begin(), diffs.end()) - diffs.begin()]; // top-right
    ordered[3] = corners[max_element(diffs.begin(), diffs.end()) - diffs.begin()]; // bottom-left
    return ordered;
}

// Detect and crop the sudoku board
static cv::Mat cropSudokuBoard(const cv::Mat &src, cv::Mat &edgeOverlay) {
    cv::Mat binary = convertToBinary(src);

    vector<vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    vector<cv::Point2f> bestCorners;
    double bestArea = 0.0;
    for (const auto &contour : contours) {
        vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);

        if (approx.size() == 4) {
            double area = cv::contourArea(approx);
            if (area > bestArea) {
                bestArea = area;
                bestCorners.clear();
                for (const auto &p : approx) {
                    bestCorners.push_back(cv::Point2f((float)p.x, (float)p.y));
                }
            }
        }
    }

    if (bestCorners.size() != 4) {
        cout << "Error: Could not find a rectangular sudoku board." << endl;
        return src.clone();
    }

    vector<cv::Point2f> orderedCorners = orderCorners(bestCorners);

    // Draw detected contour lines on the edge overlay for visualization
    edgeOverlay = src.clone();
    for (int i=0; i<orderedCorners.size(); ++i) {
        cv::line(edgeOverlay, orderedCorners[i], orderedCorners[(i + 1) % orderedCorners.size()], cv::Scalar(0, 0, 255), 3);
    }

    int side = (int)sqrt(bestArea);
    vector<cv::Point2f> dstPts = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f((float)side, 0.0f),
        cv::Point2f((float)side, (float)side),
        cv::Point2f(0.0f, (float)side)
    };

    cv::Mat transform = cv::getPerspectiveTransform(orderedCorners, dstPts);
    cv::Mat cropped;
    cv::warpPerspective(src, cropped, transform, cv::Size(side, side));
    return cropped;
}

// Extract and normalize the digit from a cell image
static cv::Mat normalizeDigit(const cv::Mat &digitBinary, int side = 20) {
    vector<vector<cv::Point>> contours;
    cv::findContours(digitBinary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return cv::Mat::zeros(side, side, CV_8U);
    }

    double bestArea = 0.0;
    int bestIdx = -1;
    for (int i = 0; i < (int)contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            bestIdx = i;
        }
    }

    if (bestIdx < 0 || bestArea < 20.0) {
        return cv::Mat::zeros(side, side, CV_8U);
    }

    cv::Rect box = cv::boundingRect(contours[bestIdx]);
    box.x = max(0, box.x - 2);
    box.y = max(0, box.y - 2);
    box.width = min(digitBinary.cols - box.x, box.width + 4);
    box.height = min(digitBinary.rows - box.y, box.height + 4);

    cv::Mat roi = digitBinary(box).clone();
    int maxDim = max(roi.cols, roi.rows);
    cv::Mat square = cv::Mat::zeros(maxDim, maxDim, CV_8U);
    int x = (maxDim - roi.cols) / 2;
    int y = (maxDim - roi.rows) / 2;
    roi.copyTo(square(cv::Rect(x, y, roi.cols, roi.rows)));

    cv::Mat resized;
    cv::resize(square, resized, cv::Size(side, side), 0, 0, cv::INTER_AREA);
    return resized;
}

// Convert the normalized digit image to a feature vector and prepare training data
static cv::Mat digitToFeatureRow(const cv::Mat &digitBinary) {
    cv::Mat normalized = normalizeDigit(digitBinary, 20);
    cv::Mat floatImg;
    normalized.convertTo(floatImg, CV_32F, 1.0 / 255.0);
    return floatImg.reshape(1, 1);
}

// Load training samples from digit images, apply augmentations, and prepare the training data and labels
static cv::Mat loadTrainingSamples(cv::Mat &labels) {
    vector<cv::Mat> samples;
    vector<int> sampleLabels;

    for (int d = 1; d <= 9; ++d) {
        string path = "numbers/" + to_string(d) + ".png";
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            cout << "Warning: could not load training digit from " << path << endl;
            continue;
        }

        cv::Mat bw;
        cv::threshold(img, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (cv::countNonZero(bw) > (int)(bw.total() / 2)) {
            cv::bitwise_not(bw, bw);
        }

        vector<cv::Point2f> shifts = {
            {0.0f, 0.0f}, {-2.0f, 0.0f}, {2.0f, 0.0f}, {0.0f, -2.0f}, {0.0f, 2.0f}
        };

        for (const auto &shift : shifts) {
            cv::Mat affine = (cv::Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
            cv::Mat shifted;
            cv::warpAffine(bw, shifted, affine, bw.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::Mat feature = digitToFeatureRow(shifted);
            samples.push_back(feature);
            sampleLabels.push_back(d);
        }
    }

    if (samples.empty()) {
        labels = cv::Mat();
        return cv::Mat();
    }

    cv::Mat trainData((int)samples.size(), samples[0].cols, CV_32F);
    for (int i = 0; i < (int)samples.size(); ++i) {
        samples[i].copyTo(trainData.row(i));
    }

    labels = cv::Mat((int)sampleLabels.size(), 1, CV_32S);
    for (int i = 0; i < (int)sampleLabels.size(); ++i) {
        labels.at<int>(i, 0) = sampleLabels[i];
    }

    return trainData;
}

// Train a Normal Bayes Classifier using the prepared training data and labels
static cv::Ptr<cv::ml::NormalBayesClassifier> trainClassifier() {
    cv::Mat labels;
    cv::Mat trainData = loadTrainingSamples(labels);

    cv::Ptr<cv::ml::NormalBayesClassifier> classifier = cv::ml::NormalBayesClassifier::create();
    if (!trainData.empty()) {
        classifier->train(trainData, cv::ml::ROW_SAMPLE, labels);
    }
    return classifier;
}


// ===================================================== //
// ============ PRIMARY ASSIGNEMNT FUNCTION ============ //
// ===================================================== //

vector<vector<int>> sudokuOCR(cv::Mat &image) {
    cout << "Running OCR on image..." << endl;
    vector<vector<int>> grid(9, vector<int>(9, 0));

    if (image.empty()) {
        cout << "Input image is empty." << endl;
        return grid;
    }

    cv::namedWindow("01 - Input image", cv::WINDOW_NORMAL);
    cv::imshow("01 - Input image", image);

    // Remove noice from the input image using morphological operations
    cv::Mat cleanedBoard = removeNoice(image);
    cv::namedWindow("02 - Cleaned board", cv::WINDOW_NORMAL);
    cv::imshow("02 - Cleaned board", cleanedBoard);

    // Detect board edges and crop to top-down view
    cv::Mat boardEdgeContour;
    cv::Mat croppedBoard = cropSudokuBoard(cleanedBoard, boardEdgeContour);
    cv::namedWindow("03 - Board edge contour", cv::WINDOW_NORMAL);
    cv::imshow("03 - Board edge contour", boardEdgeContour);
    cv::namedWindow("04 - Cropped board", cv::WINDOW_NORMAL);
    cv::imshow("04 - Cropped board", croppedBoard);

    // Convert cropped board to binary
    cv::Mat binaryBoard = convertToBinary(croppedBoard);
    cv::namedWindow("05 - Binary board", cv::WINDOW_NORMAL);
    cv::imshow("05 - Binary board", binaryBoard);

    // Load training data and train the classifier
    cv::Ptr<cv::ml::NormalBayesClassifier> classifier = trainClassifier();
    if (classifier.empty()) {
        cout << "Training failed: no digit samples were loaded." << endl;
        return grid;
    }
    
    // Draw grid lines on the cropped board for visualization
    cout << "Drawing grid overlay..." << endl;
    cout << "Cropped board size: " << croppedBoard.cols << "x" << croppedBoard.rows << " pixels." << endl;
    int cellW = croppedBoard.cols / 9;
    int cellH = croppedBoard.rows / 9;
    cv::Mat gridOverlayBoard = croppedBoard.clone();
    for (int i = 1; i < 9; ++i) {
        cv::Scalar color = cv::Scalar(0, 0, 255);
        int lineWidth = 3;
        int x = i * cellW;
        int y = i * cellH;
        cv::line(gridOverlayBoard, cv::Point(x, 0), cv::Point(x, croppedBoard.rows - 1), color, lineWidth);
        cv::line(gridOverlayBoard, cv::Point(0, y), cv::Point(croppedBoard.cols - 1, y), color, lineWidth);
    }
    cv::namedWindow("06 - Grid overlay", cv::WINDOW_NORMAL);
    cv::imshow("06 - Grid overlay", gridOverlayBoard);
    
    // Extract each cell, check for digit presence, and classify using the trained classifier
    cout << "Extracting and classifying digits..." << endl;
    cv::namedWindow("07 - Cell preview", cv::WINDOW_NORMAL);
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            // Extract the cell region from the binary board
            cv::Rect cellRect(col * cellW, row * cellH, cellW, cellH);
            cv::Mat cell = binaryBoard(cellRect).clone();

            // Define an inner region to check for digit presence, avoiding borders
            int borderX = max(1, cell.cols / 5);
            int borderY = max(1, cell.rows / 5);
            cv::Rect inner(borderX, borderY, max(1, cell.cols - 2 * borderX), max(1, cell.rows - 2 * borderY));
            cv::Mat innerCell = cell(inner).clone();

            // Skip if inner cell is empty
            if (cv::countNonZero(innerCell) < (innerCell.total() * 0.01)) {
                grid[row][col] = 0;
                continue;
            }

            // Predict the digit using the trained classifier
            cv::Mat sample = digitToFeatureRow(cell);
            int predicted = (int)classifier->predict(sample);
            if (predicted >= 1 && predicted <= 9) {
                grid[row][col] = predicted;
            } else {
                grid[row][col] = 0;
            }

            // Visualize the cell and prediction
            cv::cvtColor(cell, cell, cv::COLOR_GRAY2BGR);
            cv::rectangle(cell, inner, cv::Scalar(0, 255, 0), 1);
            cv::putText(cell, to_string(grid[row][col]), cv::Point(3, cell.rows - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::imshow("07 - Cell preview", cell);
            cv::waitKey(0);
        }
    }

    cout << "OCR Compleate. Press eneter to view results..." << endl;
    cin.get();
    return grid;
}


// =================================================== //
// ========== ORIGINAL ASSIGNMENT FUNCTIONS ========== //
// =================================================== //

vector<vector<int>> sudokuExampleGrid() {
    //Fill 9x9 grid with arbitrary points.
    //Remove
    vector<vector<int>> numbers;
    for(int i=0; i<9; i++){
        vector<int> row;
        for(int j=0; j<9; j++){
            int arbitraryNumber = (i+j)%9;
            row.push_back(arbitraryNumber);
        }
        numbers.push_back(row);
    }
    return numbers;
}

void printSudoku(vector<vector<int>> &numbers) {
    cout << "Printing sudoku:"<< endl;
    string line = "-------------------------\n";
    for (int r=0; r<numbers.size(); r++) {
        if(r%3==0)
            cout << line;
        vector<int> row = numbers[r];
        for (int c=0; c<row.size(); c++) {
            if(c%3 == 0)
                cout << "| ";
            int n = row[c];
            if (n != 0)
                cout << n << " ";
            else
                cout << ". ";
        }
        cout << "|" <<endl;
    }
    cout << line << endl;
    cout << endl;
}

/** Main method for running the Sudoku OCR test code. Set NO_IMAGE=true to test printing of a 9x9 grid
    Set NO_IMAGE=false to load the sudoku image and print the detected numbers in the grid
 */
int main(int argc, char* argv[]) {
    bool NO_IMAGE = false;

    vector<vector<int>> numbers;
    if (NO_IMAGE) {
        numbers = sudokuExampleGrid();
    } else {
        if (argc != 2) {
            cout << "Usage: ./main <imageFile>" << endl;
            return -1;
        }
        cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
        //cv::namedWindow("Input image", cv::WINDOW_FULLSCREEN);
        //cv::imshow("Input image", src);
        //cv::waitKey(0);

        numbers = sudokuOCR(src);
    }

    printSudoku(numbers);
}
