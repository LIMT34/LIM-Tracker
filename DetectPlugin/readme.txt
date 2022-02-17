
PluginSegmentation2.java
	Segmentation plugin using OpenCV


1. Creating a plugin
	Create a JavaProject in Eclipse and register the source code.
	Open the right-click menu in the left tree of Eclipse and select Export/Java/JAR file.

	In the JAR File Specification screen
	Check the "Export generated class files and resources" checkbox.
	Enter the destination (e.g., E:/DetectPlugin.jar) in the JAR file: field and click Next & Next.

	In the JAR Manifest Specification screen
	Select "Use existing manifest from workspace" and click "Next".
	Select MANIFEST.MF (/SamplePlugin/MANIFEST.MF) with the following contents created beforehand, click Finish.
	----------------------------------------------
	Manifest-Version: 1.0
	Class-Path: . /opencv-430.jar
	Plugin-Class: CellTrack.PluginSegmentation2
	----------------------------------------------

	Generate E:/DetectPlugin.jar.


2. Usage
	Copy opencv_java430.dll into C:/Fiji/lib.
	Copy the following JAR file into C:/Fiji/plugins/LIMTrackerPluginExt/plugin folder
		opencv-430.jar
		DetectPlugin.jar
