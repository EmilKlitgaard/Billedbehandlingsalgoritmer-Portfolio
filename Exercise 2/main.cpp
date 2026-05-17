#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>

using namespace std;


// ==================================================== //
// ================= HELPER FUNCTIONS ================= //
// ==================================================== //

static const int DIGIT_FEATURE_SIDE = 30;

// Helper function to display the board.
static void showBoard(const string &name, const cv::Mat &board) {
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, board);
    cv::waitKey(0);
}

// Remove noise from a binary image using morphological operations.
static cv::Mat removeNoise(const cv::Mat &src) {
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

    // Segment the board using adaptive thresholding
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 31, 7);
    
    return binary;
}

// Draw a red grid overlay on the cropped board for visualization
static cv::Mat drawGridOverlay(const cv::Mat &croppedBoard) {
    cout << "Drawing grid overlay..." << endl;
    cout << "Cropped board size: " << croppedBoard.cols << "x" << croppedBoard.rows << " pixels." << endl;
    int cellW = croppedBoard.cols / 9;
    int cellH = croppedBoard.rows / 9;

    // If croppedBoard is binary, convert to color for visualization
    cv::Mat gridOverlayBoard = croppedBoard.clone();
    if (gridOverlayBoard.channels() == 1) {
        cv::cvtColor(gridOverlayBoard, gridOverlayBoard, cv::COLOR_GRAY2BGR);
    }
    
    for (int i = 1; i < 9; ++i) {
        cv::Scalar color = cv::Scalar(0, 0, 255);
        int lineWidth = 3;
        int x = i * cellW;
        int y = i * cellH;
        cv::line(gridOverlayBoard, cv::Point(x, 0), cv::Point(x, croppedBoard.rows - 1), color, lineWidth);
        cv::line(gridOverlayBoard, cv::Point(0, y), cv::Point(croppedBoard.cols - 1, y), color, lineWidth);
    }
    return gridOverlayBoard;
}

// Detect and crop the sudoku board from a cleaned binary mask.
static cv::Mat cropSudokuBoard(const cv::Mat &src, cv::Mat &binaryMask, cv::Mat &edgeOverlay) {
    vector<vector<cv::Point>> contours;
    cv::findContours(binaryMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find the largest rectangular contour
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

    // If no corners were found, return the original image
    if (bestCorners.empty()) {
        return src.clone();
    }

    // Use RotatedRect for cleaner corner manipulation
    vector<cv::Point> contourPoints;
    for (const auto& pt : bestCorners) {
        contourPoints.push_back(cv::Point((int)pt.x, (int)pt.y));
    }
    cv::RotatedRect boardRect = cv::minAreaRect(contourPoints);
    
    // Shrink the rectangle by inset amount
    const int INSET = 8;
    boardRect.size.width = max(1.0f, boardRect.size.width - 2 * INSET);
    boardRect.size.height = max(1.0f, boardRect.size.height - 2 * INSET);
    
    // Get the inset corners
    cv::Point2f rectCorners[4];
    boardRect.points(rectCorners);
    
    // Convert to vector for consistency
    vector<cv::Point2f> orderedCorners(rectCorners, rectCorners + 4);
    
    // Sort corners: top-left, top-right, bottom-right, bottom-left
    cv::Point2f center = boardRect.center;
    sort(orderedCorners.begin(), orderedCorners.end(), [&center](const cv::Point2f& a, const cv::Point2f& b) {
        float angleA = atan2(a.y - center.y, a.x - center.x);
        float angleB = atan2(b.y - center.y, b.x - center.x);
        return angleA < angleB;
    });

    // Draw the inset board outline on the overlay for visualization
    edgeOverlay = src.clone();
    for (int i = 0; i < 4; ++i) {
        cv::line(edgeOverlay, orderedCorners[i], orderedCorners[(i + 1) % 4], cv::Scalar(0, 0, 255), 3);
    }

    // Compute output size
    float maxDim = max(boardRect.size.width, boardRect.size.height);
    int side = max(1, (int)round(maxDim));

    vector<cv::Point2f> dstPts = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f((float)side, 0.0f),
        cv::Point2f((float)side, (float)side),
        cv::Point2f(0.0f, (float)side)
    };

    cv::Mat transform = cv::getPerspectiveTransform(orderedCorners, dstPts);
    cv::Mat cropped;
    cv::warpPerspective(binaryMask, cropped, transform, cv::Size(side, side));
    return cropped;
}

// Extract and normalize the digit from a cell image
static cv::Mat normalizeDigit(const cv::Mat &digitBinary, int side = DIGIT_FEATURE_SIDE) {
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

// Isolate the center-most cluster to remove edge noise and focus on the actual digit
static cv::Mat isolateCenterCluster(const cv::Mat &digitBinary) {
    // Find all connected components
    cv::Mat labels;
    int numLabels = cv::connectedComponents(digitBinary, labels);
    
    if (numLabels <= 1) {
        return digitBinary.clone(); // Only background or one component
    }
    
    // Calculate centroid of each cluster
    cv::Point2f imageCenter(digitBinary.cols / 2.0f, digitBinary.rows / 2.0f);
    float minDistance = FLT_MAX;
    int centerClusterLabel = -1;
    
    for (int label=1; label<numLabels; ++label) {
        cv::Mat mask = (labels == label);
        cv::Moments m = cv::moments(mask);
        if (m.m00 == 0) continue; // Skip empty components
        
        cv::Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
        float distance = cv::norm(centroid - imageCenter);
        
        if (distance < minDistance) {
            minDistance = distance;
            centerClusterLabel = label;
        }
    }
    
    if (centerClusterLabel == -1) {
        return digitBinary.clone();
    }
    
    // Create binary image with only the center cluster (preserve original foreground)
    cv::Mat mask = (labels == centerClusterLabel);
    cv::Mat result = cv::Mat::zeros(digitBinary.size(), CV_8U);
    digitBinary.copyTo(result, mask);
    return result;
}

// Normalize the digit image and convert it to a feature row for classification
static cv::Mat digitToFeatureRow(const cv::Mat &digitBinary) {
    cv::Mat normalized = normalizeDigit(digitBinary, DIGIT_FEATURE_SIDE);
    cv::Mat floatImg;
    normalized.convertTo(floatImg, CV_32F, 1.0 / 255.0);
    return floatImg.reshape(1, 1);
}

// Load training samples from digit images and prepare the training data and labels.
static cv::Mat loadTrainingSamples(cv::Mat &labels) {
    vector<cv::Mat> samples;
    vector<int> sampleLabels;

    for (int digit=1; digit<=9; ++digit) {
        string path = "numbers/" + to_string(digit) + ".png";
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

        cv::Mat feature = digitToFeatureRow(bw);
        samples.push_back(feature);
        sampleLabels.push_back(digit);
    }

    if (samples.empty()) {
        labels = cv::Mat();
        return cv::Mat();
    }

    cv::Mat trainData((int)samples.size(), samples[0].cols, CV_32F);
    for (int i=0; i<(int)samples.size(); ++i) {
        samples[i].copyTo(trainData.row(i));
    }

    labels = cv::Mat((int)sampleLabels.size(), 1, CV_32S);
    for (int i=0; i<(int)sampleLabels.size(); ++i) {
        labels.at<int>(i, 0) = sampleLabels[i];
    }

    return trainData;
}

// Train a Normal Bayes Classifier using the prepared training data and labels
static cv::Ptr<cv::ml::NormalBayesClassifier> trainClassifier() {
    cv::Mat labels;
    cv::Mat trainData = loadTrainingSamples(labels);

    if (trainData.empty() || labels.empty()) {
        return cv::Ptr<cv::ml::NormalBayesClassifier>();
    }

    cv::Ptr<cv::ml::NormalBayesClassifier> classifier = cv::ml::NormalBayesClassifier::create();
    if (!classifier->train(trainData, cv::ml::ROW_SAMPLE, labels)) {
        return cv::Ptr<cv::ml::NormalBayesClassifier>();
    }

    return classifier;
}


// ===================================================== //
// ============ PRIMARY ASSIGNEMNT FUNCTION ============ //
// ===================================================== //

vector<vector<int>> sudokuOCR(cv::Mat &image) {
    // Load training data and train the classifier
    cout << "Loading training data and training classifier..." << endl;
    cv::Ptr<cv::ml::NormalBayesClassifier> classifier = trainClassifier();
    vector<vector<int>> grid(9, vector<int>(9, 0));
    if (classifier.empty()) {
        cout << "Training failed: no digit samples were loaded." << endl;
        return grid;
    }
    cout << "Training complete. Starting OCR process..." << endl;

    // Display the original input image
    showBoard("01 - Input image", image);

    // Convert to binary first, then clean noise in the binary domain.
    cv::Mat binaryImage = convertToBinary(image);
    showBoard("02 - Binary board", binaryImage);
    cv::Mat cleanedBoard = removeNoise(binaryImage);
    showBoard("03 - Cleaned board", cleanedBoard);

    // Detect board edges and crop to a top-down view.
    cv::Mat boardEdgeContour;
    cv::Mat croppedBoard = cropSudokuBoard(image, cleanedBoard, boardEdgeContour);
    showBoard("04 - Board edge contour", boardEdgeContour);
    showBoard("05 - Cropped board", croppedBoard);

    // Draw a grid overlay on the cropped board for visualization.
    cv::Mat gridOverlayBoard = drawGridOverlay(croppedBoard);
    showBoard("06 - Grid overlay", gridOverlayBoard);

    // Extract each cell, check for digit presence, and classify using the trained classifier.
    cout << "Extracting and classifying digits..." << endl;
    int cellW = croppedBoard.cols / 9;
    int cellH = croppedBoard.rows / 9;
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            // Extract the cell region from the binary board
            cv::Rect cellRect(col * cellW, row * cellH, cellW, cellH);
            cv::Mat cell = croppedBoard(cellRect).clone();

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

            // Isolate the center-most cluster to remove edge noise
            cv::Mat centeredCell = isolateCenterCluster(cell);

            // Predict the digit using the trained Bayesian classifier.
            int predicted = 0;

            cv::Mat sample = digitToFeatureRow(centeredCell);
            predicted = classifier->predict(sample);

            // Only accept predictions in the valid digit range (1-9)
            if (!(predicted >= 1 && predicted <= 9)) {
                continue;
            }

            // Display the cell with the predicted digit for visualization
            cv::Mat cellPreview;
            cv::cvtColor(centeredCell, cellPreview, cv::COLOR_GRAY2BGR);
            cv::rectangle(cellPreview, inner, cv::Scalar(0, 255, 0), 1);
            cv::putText(cellPreview, to_string(predicted), cv::Point(3, cellPreview.rows - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            showBoard("07 - Cell preview", cellPreview);

            grid[row][col] = predicted;
        }
    }

    cout << "OCR Complete. Press enter to view results..." << endl;
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
