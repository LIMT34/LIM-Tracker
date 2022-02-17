
PluginDeepLearning2.java
	Plugin for integration with deep learning recognition function.


1. Creating a plugin
	Create a JavaProject in Eclipse and register the source code.
	Open the right-click menu in the left tree of Eclipse and select Export/Java/JAR file.

	In the JAR File Specification screen
	Check the "Export generated class files and resources" checkbox.
	Enter the destination (e.g., E:/DLLinkPlugin.jar) in the JAR file: field and click Next & Next.

	In the JAR Manifest Specification screen
	Select "Use existing manifest from workspace" and click "Next".
	Select MANIFEST.MF (/SamplePlugin/MANIFEST.MF) with the following contents created beforehand, click Finish.
	----------------------------------------------
	Manifest-Version: 1.0
	Plugin-Class: CellTrack.PluginDeepLearning2
	----------------------------------------------

	Generate E:/DLLinkPlugin.jar.


2. Usage
	Copy the DLLinkPlugin.jar into C:/Fiji/plugins/LIMTrackerPluginExt/plugin folder.
