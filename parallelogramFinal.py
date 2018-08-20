# Import required modules
import numpy as np
import cv2
import pdb
import math
import argparse

class DetectParallelogram(object):
    def __init__(self):
        '''
        Command Line arguments:

        --testImage <absoulte/relative path> => Ex "TestImage1c.png"
            -> Can be used to specify input image.
        --sobelThreshold <threshold> => Ex "100"
            -> Can be used to specify threshold for sobel edge detection.
        '''
        # Parse command line Arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--testImage", dest="testImage", required=True, help="Enter path of image to be processed")
        parser.add_argument("--sobelThreshold", dest="sobelThreshold", default="100", help="Enter path of image to be processed")
        parser.add_argument("--houghProb", dest="houghProb", default="0.7",
                            help="Probability of being a line after hough accumulator voting")
        parser.add_argument("--dupeLineThresh", dest="dupeLineThresh", default="20",
                            help="Pixels width which will be considered as single line for hough line mapping")
        self.testArgs = parser.parse_args()

        # Initializing all images that will be used
        self.inpImg = cv2.imread(self.testArgs.testImage)
        self.refRows = self.inpImg.shape[0]
        self.refCols = self.inpImg.shape[1]
        self.grayImg = np.zeros((self.refRows, self.refCols), dtype=np.uint8)
        self.sobelImg = np.zeros((self.refRows, self.refCols), dtype=np.uint8)
        self.edgeImg = np.zeros((self.refRows, self.refCols), dtype=np.uint8)
        self.houghLines = np.zeros((self.refRows, self.refCols), dtype=np.uint8)

    def convertColorToGray(self):
        '''
        This function converts input colored image to gray scale image.
        '''
        print "Color image to gray scale conversion started"

        # Seperating R, G, B matrices from color image
        redMat = self.inpImg[:, :, 2]
        blueMat = self.inpImg[:, :, 0]
        greenMat = self.inpImg[:, :, 1]

        # Gray = 0.30R + 0.59G + 0.11B
        print "Calculating luminousity using formula: L = 0.3R + 0.59G + 0.11B"
        for row in range(0, self.refRows):
            for col in range(0, self.refCols):
                lum = int(0.30 * redMat[row][col] + 0.59 * greenMat[row][col] + 0.11 * blueMat[row][col])
                self.grayImg[row][col] = lum
        print "Conversion from color to gray image completed."

    def detectEdges(self):
        '''
        This function detects edges by calculating sobel and thresholding it
        '''
        print "Detecting edges starting now with threshold: {0}".format(self.testArgs.sobelThreshold)
        # Zero padded image creation
        zeroPaddedGray = np.pad(self.grayImg, (1, 1), 'constant', constant_values=0)

        for row in range(0, self.refRows):
            for col in range(0, self.refCols):
                sx = zeroPaddedGray[row, col] * (-1) + zeroPaddedGray[row + 1, col] * (-2) + zeroPaddedGray[
                                                                                                 row + 2, col] * (-1) + \
                     zeroPaddedGray[
                         row, col + 2] * (1) + zeroPaddedGray[row + 1, col + 2] * (2) + zeroPaddedGray[
                                                                                            row + 2, col + 2] * (1)
                sy = zeroPaddedGray[row + 2, col] * (-1) + zeroPaddedGray[row + 2, col + 1] * (-2) + zeroPaddedGray[
                                                                                                         row + 2, col + 2] * (
                                                                                                     -1) + \
                     zeroPaddedGray[
                         row, col] * (1) + zeroPaddedGray[row, col + 1] * (2) + zeroPaddedGray[row, col + 2] * (1)
                edgeVal = np.sqrt(sx ** 2 + sy ** 2)
                self.sobelImg[row, col] = edgeVal
                self.edgeImg[row, col] = 0 if edgeVal < int(self.testArgs.sobelThreshold) else 255
        print "Detecting edges completed"

    def _calSinCosTheta(self, thetaList):
        '''
        This function can be used to cached values for sin(theta) and cos(theta)
        Can not be called outside of this class
        :param thetaList: List of Theta in radians
        :return:
            sinThetaList, colThetaList
        '''
        sinThetaList = []
        cosThetaList = []
        for theta in thetaList:
            sinThetaList.append(math.sin(theta))
            cosThetaList.append(math.cos(theta))
        return sinThetaList, cosThetaList

    def calImgHoughTransform(self):
        '''
        This function calculates hough transform of image
        '''
        print "Starting hough transform"
        rho = np.sqrt(self.refRows ** 2 + self.refCols ** 2)
        thetas = np.radians(np.arange(-90, 90))
        print "max distance hough line: {0}".format(rho)
        self.houghSpaceDisp = np.zeros((len(thetas), int(2 * np.ceil(rho))), dtype=np.uint8)
        self.houghSpace = np.zeros((len(thetas), int(2 * np.ceil(rho))), dtype=np.uint8)
        print "Hough Space dimensions (rows, cols): {0}".format(self.houghSpaceDisp.shape)

        # why to calculate again n again
        sinThetaDict, cosThetaDict = self._calSinCosTheta(thetaList=thetas)

        for row in range(0, self.refRows):
            for col in range(0, self.refCols):
                if self.edgeImg[row, col] == 255:  # Try hough transform on only white pixels after edge detection
                    index = 0
                    for theta in thetas:
                        try:
                            rad = round((col * cosThetaDict[index] + row * sinThetaDict[index]))
                            if self.houghSpaceDisp[int(np.degrees(theta)) + 90, int(rad + int((np.ceil(rho))))] < 255:
                                self.houghSpaceDisp[int(np.degrees(theta)) + 90, int(rad + int((np.ceil(rho))))] += 1
                            index += 1
                        except Exception as e:
                            print "Exception occured while execution: {0}".format(e)
                            print "Important info to debug"
                            print "Row: {0}\nCol: {1}".format(row, col)
                            raise Exception(e)

        # Move this image to other image to see hough transform at this point
        for row in range(self.houghSpaceDisp.shape[0]):
            for col in range(self.houghSpaceDisp.shape[1]):
                self.houghSpace[row, col] = self.houghSpaceDisp[row][col]

        maxVote = self.houghSpace.max()
        houghProb = self.houghSpace / float(maxVote)

        self.lineInfo = {}
        lineNo = 1
        for row in range(houghProb.shape[0]):
            for col in range(houghProb.shape[1]):
                if houghProb[row, col] > float(self.testArgs.houghProb):
                    self.houghSpace[row, col] = 255
                    self.lineInfo["".join(("line", str(lineNo)))] = {}
                    self.lineInfo["".join(("line", str(lineNo)))]["rho"] = col - int((np.ceil(rho)))
                    self.lineInfo["".join(("line", str(lineNo)))]["theta"] = row - 90
                    lineNo += 1
                else:
                    self.houghSpace[row, col] = 0
        drawLineInfo = self._houghToImgSpaceConv()
        self._drawHoughLinesOnGray(drawLineDict=drawLineInfo)

    def _houghToImgSpaceConv(self):
        for line in self.lineInfo:
            if self.lineInfo[line]["theta"] == 0 or self.lineInfo[line]["theta"] == 180:
                self.lineInfo[line]["special"] = "hor"
            elif self.lineInfo[line]["theta"] == 90 or self.lineInfo[line]["theta"] == -90:
                self.lineInfo[line]["special"] = "ver"
            else:
                self.lineInfo[line]["slope"] = -1 / (math.tan(np.radians(self.lineInfo[line]["theta"])))
                self.lineInfo[line]["const"] = self.lineInfo[line]["rho"] / math.sin(
                    np.radians(self.lineInfo[line]["theta"]))

        # Calculating max end points of lines in image
        for line in self.lineInfo:
            fx = 0
            fy = 0
            sx = 0
            sy = 0
            preX = 0
            preY = 0
            for x in range(self.refCols):
                if "special" in self.lineInfo[line].keys():
                    continue
                    # if self.lineInfo[line]["special"] == "hor":
                    #     pass
                    # else:
                    #     pass
                else:
                    y = x * self.lineInfo[line]["slope"] + self.lineInfo[line]["const"]
                    if fx == 0 and round(y) in range(0, self.refRows):
                        fx = x
                        fy = y
                        preX = x
                        preY = y
                        continue
                    if fx != 0:
                        if round(y) in range(0, self.refRows):
                            preY = y
                            preX = x
                        else:
                            sx = preX
                            sy = preY
                if sx == 0 and sy == 0:
                    y = (self.refCols - 1 - x) * self.lineInfo[line]["slope"] + self.lineInfo[line]["const"]
                    if round(y) in range(0, self.refRows):
                        sx = (self.refCols - 1 - x)
                        sy = y

            self.lineInfo[line]["fx"] = int(np.floor(fx))
            self.lineInfo[line]["fy"] = int(np.floor(fy))
            self.lineInfo[line]["sx"] = int(np.floor(sx))
            self.lineInfo[line]["sy"] = int(np.floor(sy))

        uniqueLineInfo = {}
        for line in self.lineInfo:
            if "special" in self.lineInfo[line].keys():
                continue
            if not uniqueLineInfo:
                uniqueLineInfo[line] = {}
                uniqueLineInfo[line] = self.lineInfo[line]
            else:
                remInfo = False
                for prevLine in uniqueLineInfo:
                    if abs(uniqueLineInfo[prevLine]["fx"] - self.lineInfo[line]["fx"]) < int(self.testArgs.dupeLineThresh) and abs(
                                    uniqueLineInfo[prevLine]["fy"] - self.lineInfo[line]["fy"]) < int(self.testArgs.dupeLineThresh) and abs(
                                uniqueLineInfo[prevLine]["sx"] - self.lineInfo[line]["sx"]) < int(self.testArgs.dupeLineThresh) and abs(
                                uniqueLineInfo[prevLine]["sy"] - self.lineInfo[line]["sy"]) < int(self.testArgs.dupeLineThresh):
                        print "Dupe Line detected....REMOVING!!!!"
                        remInfo = True
                        break
                if not remInfo:
                    uniqueLineInfo[line] = {}
                    uniqueLineInfo[line] = self.lineInfo[line]
        return uniqueLineInfo

    def _drawHoughLinesOnGray(self, drawLineDict):
        for line in drawLineDict:
            if "special" in drawLineDict[line].keys():
                continue
            cv2.line(self.houghLines, (drawLineDict[line]["fx"], drawLineDict[line]["fy"]),
                     (drawLineDict[line]["sx"], drawLineDict[line]["sy"]), (255, 255, 255), 2)

    def displayAllImages(self):
        '''
        Displays all images
        '''
        print "Listing all images to be displayed"
        print "1. Input Image"
        print "2. Gray Scale Image"
        print "3. Sobel Image"
        print "4. Edge Detected Image"
        print "5. Edges in Hough Space"
        print "6. Grayscale image with hough lines"
        scaledRow = int(self.refRows * 0.40)
        scaledCol = int(self.refCols * 0.40)
        cv2.namedWindow('Input_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Input_image', scaledCol,scaledRow)
        cv2.namedWindow('Gray_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gray_image', scaledCol, scaledRow)
        cv2.namedWindow('Sobel_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sobel_image', scaledCol, scaledRow)
        cv2.namedWindow('Edge_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Edge_image', scaledCol, scaledRow)
        cv2.namedWindow('Hough_space', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hough_space', 900, 300)
        cv2.namedWindow('Gray_hough_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gray_hough_image', scaledCol, scaledRow)
        cv2.imshow("Input_image", self.inpImg)
        cv2.imshow("Gray_image", self.grayImg)
        cv2.imshow("Sobel_image", self.sobelImg)
        cv2.imshow("Edge_image", self.edgeImg)
        cv2.imshow("Hough_space", self.houghSpaceDisp)
        cv2.imshow("Gray_hough_image", self.houghLines)
        cv2.waitKey(0)


####################################################################################################
###################################### EXECUTION FLOW BEGINS HERE##################################
####################################################################################################

step = DetectParallelogram()
step.convertColorToGray()
step.detectEdges()
step.calImgHoughTransform()
step.displayAllImages()
