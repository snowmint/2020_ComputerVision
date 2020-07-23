#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {

	// Create a VideoCapture object and use camera to capture the video
	VideoCapture cap(0);
	// Check if camera opened successfully
	if (!cap.isOpened())
	{
		cout << "Error opening video stream" << endl;
		return -1;
	}

	// Default resolution of the frame is obtained.The default resolution is system dependent. 

	int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

	// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file. 
	//motion jpg
	VideoWriter video("outcpp.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 24, Size(frame_width, frame_height));
	while (1)
	{
		Mat frame;
		// Capture frame-by-frame 
		cap >> frame;
		// If the frame is empty, break immediately
		if (frame.empty())
			break;
		// Write the frame into the file 'outcpp.avi'
		video.write(frame);
		// Display the resulting frame    
		imshow("Frame", frame);
		// Press  ESC on keyboard to  exit
		char c = (char)waitKey(1);
		if (c == 27)
			break;
	}

	// When everything done, release the video capture and write object

	cap.release();
	video.release();
	// Closes all the windows
	destroyAllWindows();
	return 0;
}


/*
int main(int argc, char* argv[])
{
	//open the video file for reading
	VideoCapture cap("D:/A Herd of Deer Running.mp4");

	// if not success, exit program
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video file" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	//Uncomment the following line if you want to start the video in the middle
	//cap.set(CAP_PROP_POS_MSEC, 300); 

	//get the frames rate of the video
	double fps = cap.get(CAP_PROP_FPS);
	cout << "Frames per seconds : " << fps << endl;

	String window_name = "My First Video";

	namedWindow(window_name, WINDOW_NORMAL); //create a window

	while (true)
	{
		Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video 

										 //Breaking the while loop at the end of the video
		if (bSuccess == false)
		{
			cout << "Found the end of the video" << endl;
			break;
		}

		//show the frame in the created window
		imshow(window_name, frame);

		//wait for for 10 ms until any key is pressed.  
		//If the 'Esc' key is pressed, break the while loop.
		//If the any other key is pressed, continue the loop 
		//If any key is not pressed withing 10 ms, continue the loop
		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}
	}

	return 0;

}*/

/*
int main(int argc, char* argv[])
{
	//Open the default video camera
	VideoCapture cap(0);

	// if not success, exit program
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video camera" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Resolution of the video : " << dWidth << " x " << dHeight << endl;

	string window_name = "My Camera Feed";
	namedWindow(window_name); //create a window called "My Camera Feed"

	while (true)
	{
		Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video 

										 //Breaking the while loop if the frames cannot be captured
		if (bSuccess == false)
		{
			cout << "Video camera is disconnected" << endl;
			cin.get(); //Wait for any key press
			break;
		}

		//show the frame in the created window
		imshow(window_name, frame);

		//wait for for 10 ms until any key is pressed.  
		//If the 'Esc' key is pressed, break the while loop.
		//If the any other key is pressed, continue the loop 
		//If any key is not pressed withing 10 ms, continue the loop 
		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}
	}

	return 0;

}
*/




/*
int main()
{
    string imageName = "test.png";

	Mat image;
	image = imread(imageName, 1);

	// Mat gray_image;
	//cvtColor(image, gray_image, CV_RGB2GRAY);

	//imwrite("maple girl gray.jpg", gray_image);

	//namedWindow(imageName, CV_WINDOW_AUTOSIZE);
	//namedWindow("Gray image", CV_WINDOW_AUTOSIZE);

	imshow(imageName, image);
	//imshow("Gray image", gray_image);

	waitKey(0);

	return 0;
}*/