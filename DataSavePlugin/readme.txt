
PluginSaveData2.java
	Plugin for displaying ROI information in the console.
	
1. Creating a plugin
	Create a JavaProject in Eclipse and register the source code.
	Open the right-click menu in the left tree of Eclipse and select Export/Java/JAR file.

	In the JAR File Specification screen
	Check the "Export generated class files and resources" checkbox.
	Enter the destination (e.g., E:/SaveDataTestPlugin.jar) in the JAR file: field and click Next & Next.

	In the JAR Manifest Specification screen
	Select "Use existing manifest from workspace" and click "Next".
	Select MANIFEST.MF (/SamplePlugin/MANIFEST.MF) with the following contents created beforehand, click Finish.
	----------------------------------------------
	Manifest-Version: 1.0
	Plugin-Class: CellTrack.PluginSaveData2
	----------------------------------------------

	Generate E:/SaveDataTestPlugin.jar.


2. Usage
	Copy the following JAR file into C:/Fiji/plugins/LIMTrackerPluginExt/plugin folder
		SaveDataTestPlugin.jar
