#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


// ==================================================== //
// ================= HELPER FUNCTIONS ================= //
// ==================================================== //

static const int DIGIT_FEATURE_SIDE = 64;

// Try to load an ONNX CNN model for digit recognition. Returns empty Net if not found.
static cv::dnn::Net loadDigitCNN(const std::string &path) {
    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(path);
        if (net.empty()) return cv::dnn::Net();
        return net;
    } catch (const cv::Exception &) {
        return cv::dnn::Net();
    }
}

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

    // Segment the board using adaptive thresholding
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

    // Show labels for debugging
    /*cv::Mat labelVis;
    cv::normalize(labels, labelVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(labelVis, labelVis, cv::COLORMAP_JET);
    cv::namedWindow("Connected Components", cv::WINDOW_NORMAL);
    cv::imshow("Connected Components", labelVis);
    cv::waitKey(0);*/
    
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

// Build CNN input with the same normalization used during PyTorch training:
// ToTensor() + Normalize((0.5,), (0.5,))  =>  (x - 0.5) / 0.5  ==  x * 2 - 1
static cv::Mat makeCNNInputBlob(const cv::Mat &digitBinary) {
    cv::Mat normalized = normalizeDigit(digitBinary, DIGIT_FEATURE_SIDE);
    return cv::dnn::blobFromImage(
        normalized,
        1.0 / 127.5,
        cv::Size(DIGIT_FEATURE_SIDE, DIGIT_FEATURE_SIDE),
        cv::Scalar(127.5),
        false,
        false,
        CV_32F
    );
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

        // Enhanced augmentation with translations, rotations, scaling, and morphological variations
        vector<cv::Point2f> shifts = {
            {0.0f, 0.0f}, {-3.0f, 0.0f}, {3.0f, 0.0f}, {0.0f, -3.0f}, {0.0f, 3.0f},
            {-2.0f, -2.0f}, {2.0f, 2.0f}, {-2.0f, 2.0f}, {2.0f, -2.0f}
        };
        vector<float> angles = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
        vector<float> scales = {0.85f, 0.92f, 1.0f, 1.08f, 1.15f};

        for (const auto &shift : shifts) {
            for (float angle : angles) {
                for (float scale : scales) {
                    cv::Mat augmented = bw.clone();
                    
                    // Apply translation
                    cv::Mat affineShift = (cv::Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
                    cv::warpAffine(augmented, augmented, affineShift, augmented.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
                    
                    // Apply rotation and scaling
                    if (angle != 0.0f || scale != 1.0f) {
                        cv::Point2f center(augmented.cols / 2.0f, augmented.rows / 2.0f);
                        cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, scale);
                        cv::warpAffine(augmented, augmented, rotMat, augmented.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
                    }
                    
                    cv::Mat feature = digitToFeatureRow(augmented);
                    samples.push_back(feature);
                    sampleLabels.push_back(d);
                }
            }
        }

        // Add morphological variations
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::Mat eroded, dilated;
        cv::erode(bw, eroded, kernel, cv::Point(-1, -1), 1);
        cv::dilate(bw, dilated, kernel, cv::Point(-1, -1), 1);
        
        cv::Mat featureEroded = digitToFeatureRow(eroded);
        samples.push_back(featureEroded);
        sampleLabels.push_back(d);
        
        cv::Mat featureDilated = digitToFeatureRow(dilated);
        samples.push_back(featureDilated);
        sampleLabels.push_back(d);
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
    
    // Try to load a pre-trained CNN (ONNX). If present, prefer DNN inference for accuracy.
    cv::dnn::Net digitNet = loadDigitCNN("digit_cnn.onnx");
    bool useDNN = !digitNet.empty();
    if (useDNN) cout << "Loaded CNN model." << endl;
    
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

            // Isolate the center-most cluster to remove edge noise
            cv::Mat centeredCell = isolateCenterCluster(cell);

            // Predict the digit using the best available method (CNN if available, else Bayesian)
            int predicted = 0;

            if (useDNN) {
                // Prepare input with training-compatible normalization.
                cv::Mat cnnIn = normalizeDigit(centeredCell, DIGIT_FEATURE_SIDE);
                cv::Mat blob = makeCNNInputBlob(centeredCell);
                digitNet.setInput(blob);
                cv::Mat logits = digitNet.forward(); // 1 x classes
                if (!logits.empty()) {
                    cv::Mat flat = logits.reshape(1, 1).clone();
                    flat.convertTo(flat, CV_32F);
                    if (flat.total() == 9) {
                        double minVal, maxVal;
                        cv::Point minLoc, maxLoc;
                        cv::minMaxLoc(flat, &minVal, &maxVal, &minLoc, &maxLoc);
                        predicted = maxLoc.x + 1; // map 0->1, 1->2, ..., 8->9
                    } else {
                        predicted = 0;
                    }
                }

                // Debug preview from CNN input
                cv::Mat samplePreview = cnnIn;
                cv::resize(samplePreview, samplePreview, cv::Size(), 10.0, 10.0, cv::INTER_NEAREST);
                cv::namedWindow("Digit sample", cv::WINDOW_NORMAL);
                cv::imshow("Digit sample", samplePreview);
            } else {
                cv::Mat sample = digitToFeatureRow(centeredCell);
                cv::Mat samplePreview = sample.reshape(1, DIGIT_FEATURE_SIDE);
                cv::resize(samplePreview, samplePreview, cv::Size(), 10.0, 10.0, cv::INTER_NEAREST);
                cv::namedWindow("Digit sample", cv::WINDOW_NORMAL);
                cv::imshow("Digit sample", samplePreview);

                // Predict with confidence threshold using NormalBayesClassifier
                cv::Mat outputs;
                cv::Mat probs;
                classifier->predictProb(sample, outputs, probs);
                if (!outputs.empty()) {
                    // outputs may be float or int; try float
                    if (outputs.type() == CV_32F)
                        predicted = (int)outputs.at<float>(0, 0);
                    else
                        predicted = (int)outputs.at<int>(0, 0);
                }
                if (!(predicted >= 1 && predicted <= 9)) {
                    predicted = 0;
                }
            }

            // Visualize the centeredCell and prediction
            cv::cvtColor(centeredCell, centeredCell, cv::COLOR_GRAY2BGR);
            cv::rectangle(centeredCell, inner, cv::Scalar(0, 255, 0), 1);
            grid[row][col] = predicted;
            cv::putText(centeredCell, to_string(grid[row][col]), cv::Point(3, centeredCell.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::imshow("07 - Cell preview", centeredCell);
            cv::waitKey(0);
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
