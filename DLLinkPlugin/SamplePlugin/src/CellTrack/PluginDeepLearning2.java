package CellTrack;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Properties;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.SelectionAdapter;
import org.eclipse.swt.events.SelectionEvent;
import org.eclipse.swt.events.VerifyEvent;
import org.eclipse.swt.events.VerifyListener;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.graphics.Font;
import org.eclipse.swt.graphics.GC;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.widgets.Button;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.DirectoryDialog;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.FileDialog;
import org.eclipse.swt.widgets.Label;
import org.eclipse.swt.widgets.MessageBox;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.swt.widgets.Text;

public class PluginDeepLearning2 implements PluginDeepLearningIF {

    private static final String INIT_FILE_PATH1 = System.getProperty("user.dir") + "/plugins/LIMTrackerPluginExt/plugin/plugin.properties";
    private static final String INIT_FILE_PATH2 = System.getProperty("user.dir") + "/plugin/plugin.properties";
    private String filePath;
	private static Properties properties = null;
	private int DeepLearningType = -1;
	private ArrayList<String> DeepLearningName = new ArrayList<String>(); 
	private ArrayList<String> DeepLearningService = new ArrayList<String>(); 
	private int limitSize_train = 1000;
	private int limitsize_detect = 1000;
	private int imageWidth = -1;
	private int imageHeight = -1;
	
	int resnetType = -1;	//0:resnet50, 1:resnet101
	int imagesPerGPU = -1;

	private int cellposeCpuMode = 0;
	
	private int stepperepoch;
	private int epoch1;
	private int epoch0;
	private int gpu;
	private int resnet;
	private String saveFoldername;

	private Composite baseComponent = null;
	private Text text_stepperepoch_ubuntu;
	private Text text_epoch1_ubuntu;
	private Text text_gpu_ubuntu;
	private Text text_initWeightFilename;
	private Text text_saveFoldername;
	private Text text_maskFoldername;

	Button button_initWeightFilename;
	
	private Label label_stepperepoch;
    private Label label_epoch1;
    private Label label_gpu;
    
    private Button resnet101;
	private Button resnet50;

	private Label label_filename;
	private Label label_foldername;
	
	private Shell shell;
	
	private boolean overwrite = true;
	
	public String toString(){
		if(DeepLearningType < 0 || DeepLearningType > 4) {
			return "DL Segmentation Plugin";
		}else {
			return DeepLearningName.get(DeepLearningType);
		}
	}

	public ArrayList<String> getDeepLearningName(){
		return DeepLearningName;
	}
	
	public void setDeepLearningName(String deepLearningName){
		for(int i = 0; i < DeepLearningName.size(); i++) {
			if(deepLearningName.equals(DeepLearningName.get(i))){
				if(DeepLearningType == i) {
					overwrite = false;
				}else {
					DeepLearningType = i;
					overwrite = true;
				}
				break;
			}
		}
	}
	
    private String getProperty(final String key) {
        return getProperty(key, "");
    }

    private String getProperty(final String key, final String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
    
	private boolean setProperties() {

		if(overwrite) {
			overwrite = false;
		}else {
			return false;
		}
		
    	String os = System.getProperty("os.name");
		if((os.equals("Windows 7") || os.equals("Windows 10"))) {
			//ファイルパス設定
	    	filePath = INIT_FILE_PATH1;
	    	if (!new File(filePath).exists()) {
	    		filePath = INIT_FILE_PATH2;
	    	}
    	}else {
			//ファイルパス設定
	    	filePath = INIT_FILE_PATH1;
	    	filePath = filePath.replace("plugin.properties", "plugin_ubuntu.properties");
	    	if (!new File(filePath).exists()) {
	    		filePath = INIT_FILE_PATH2;
	    		filePath = filePath.replace("plugin.properties", "plugin_ubuntu.properties");
	    	}
    	}
		System.out.println("DLPluginProperties: " + filePath);
		
		try {
	    	properties = new Properties();
	    	properties.load(Files.newBufferedReader(Paths.get(filePath), StandardCharsets.UTF_8));

			try {
				if(DeepLearningType == -1) {
					int type = Integer.parseInt(getProperty("deep_learning_type"));
					if(type < 0 || type > 4) type = 1; //matterport
					DeepLearningType = type;
				}
			}catch(Exception e) {}

			try {
				DeepLearningName.add(getProperty("deep_learning_name0"));
				DeepLearningService.add(getProperty("deep_learning_service0"));
				DeepLearningName.add(getProperty("deep_learning_name1"));
				DeepLearningService.add(getProperty("deep_learning_service1"));
				DeepLearningName.add(getProperty("deep_learning_name2"));
				DeepLearningService.add(getProperty("deep_learning_service2"));
				DeepLearningName.add(getProperty("deep_learning_name3"));
				DeepLearningService.add(getProperty("deep_learning_service3"));
				DeepLearningName.add(getProperty("deep_learning_name4"));
				DeepLearningService.add(getProperty("deep_learning_service4"));

//				if (DeepLearningService.contains(".exe")) {
//					File file = new File(DeepLearningService);
//					if(!file.exists()) {
//						DeepLearningService = "";
//					}
//
//				}else if (DeepLearningService.contains(".py")) {

//				}else{
//					DeepLearningService = "";
//				}

			}catch(Exception e) {}

			try {
				if(Integer.parseInt(getProperty("resnet_type")) == 0 || Integer.parseInt(getProperty("resnet_type")) == 1){
					resnet = 101;
			    	if(Integer.parseInt(getProperty("resnet_type")) == 0) {
			    		resnet = 50;
			    	}
				}
			}catch(Exception e) {}
			try {
				if(Integer.parseInt(getProperty("images_per_gpu")) > 0) gpu = Integer.parseInt(getProperty("images_per_gpu"));
			}catch(Exception e) {}
			
			try {
				if(Integer.parseInt(getProperty("cellpose_cpu_mode")) >= 0) cellposeCpuMode = Integer.parseInt(getProperty("cellpose_cpu_mode"));
			}catch(Exception e) {}
			
			try {
				limitSize_train =Integer.parseInt(getProperty("limitsize_train"));
				if(limitSize_train < 1) limitSize_train = 1;
			}catch(Exception e) {}
			try {
				limitsize_detect =Integer.parseInt(getProperty("limitsize_detect"));
				if(limitsize_detect < 1) limitsize_detect = 1;
			}catch(Exception e) {}

			try {
				if(Integer.parseInt(getProperty("resnet_type")) == 0 || Integer.parseInt(getProperty("resnet_type")) == 1) resnetType = Integer.parseInt(getProperty("resnet_type"));
			}catch(Exception e) {}
			try {
				if(Integer.parseInt(getProperty("images_per_gpu")) > 0) imagesPerGPU = Integer.parseInt(getProperty("images_per_gpu"));
			}catch(Exception e) {}

			return true;
			
	    } catch (IOException e) {
	        System.out.println("Properties file load　ERROR: " + filePath);
			MessageBox box2 = new MessageBox(shell, SWT.OK | SWT.ICON_WARNING);
			box2.setMessage("Properties file load　ERROR: " + filePath);
			int ret2 = box2.open();
			switch(ret2){
			    case SWT.OK:
			       break;
			}
	    }
        return false;
	}
	
	public Composite getParameterSettingPanel(Composite parent, boolean largeWindow){

		if(!setProperties()) {
			//同じアルゴの場合そのまま返す
			return baseComponent;
		}
		
		if(baseComponent != null) {
			//アルゴにあわせＧＵＩ調整
			adjustGUI();
			return baseComponent;
		}
		
		shell = parent.getShell();

  		//フォントサイズ自動調整
		int fontSizeMinimize = 0;
  		int limitSize = 64;
  		if(largeWindow){
  			limitSize = 85;
  		}
  		int minimize = 0; 
  		Label label_dammy = new Label(shell, SWT.NONE);
  		for(int i = -5; i <= 5; i++){
	  		label_dammy.setFont(new Font(Display.getDefault(), "Arial", 8+i, SWT.NONE));
	  		Point textSize = new GC(label_dammy).textExtent("-------------");//Circle Ratio
	  		if(textSize.x <= limitSize){
	  			minimize = i;
	  		}else{
	  			break;
	  		}
  		}
  		fontSizeMinimize += (-minimize);

		Font fontArial = new Font(Display.getDefault(), "Arial", 8-fontSizeMinimize, SWT.NONE);
		Color color247 = new Color(Display.getDefault(), 247,247,247);
		Color color128 = new Color(Display.getDefault(), 110,110,110);
		Color color064 = new Color(Display.getDefault(), 235,235,235);
		Color colorNote = new Color(Display.getDefault(), 255, 245, 245);
	
        baseComponent = new Composite(parent, SWT.NONE);
        baseComponent.setLayout(null);

		//■StepPerEpoch
		label_stepperepoch = new Label(baseComponent, SWT.NONE);
		label_stepperepoch.setText("Step per epoch ");
		label_stepperepoch.setFont(fontArial);
		text_stepperepoch_ubuntu = new Text(baseComponent,SWT.SINGLE|SWT.BORDER | SWT.CENTER);
		text_stepperepoch_ubuntu.addVerifyListener(new VerifyListener() {
			public void verifyText(VerifyEvent ve) {
        	    if (ve.character < 0x0020) {
        	    } else if (ve.character >= 0x0030 && ve.character <= 0x0039) {
        	    } else ve.doit = false;//上記以外の文字の場合は入力拒否
			}
        });
		text_stepperepoch_ubuntu.setText(Integer.toString(50));
		text_stepperepoch_ubuntu.setFont(fontArial);

		//■Epoch1
		label_epoch1 = new Label(baseComponent, SWT.NONE);
		label_epoch1.setText("Number of epochs");
		label_epoch1.setFont(fontArial);
		text_epoch1_ubuntu = new Text(baseComponent,SWT.SINGLE|SWT.BORDER | SWT.CENTER);

		text_epoch1_ubuntu.addVerifyListener(new VerifyListener() {
			public void verifyText(VerifyEvent ve) {
        	    if (ve.character < 0x0020) {
        	    } else if (ve.character >= 0x0030 && ve.character <= 0x0039) {
        	    } else ve.doit = false;//上記以外の文字の場合は入力拒否
			}
        });
		text_epoch1_ubuntu.setText(Integer.toString(400));
		text_epoch1_ubuntu.setFont(fontArial);

		//■RESNET
		resnet101 = new Button(baseComponent, SWT.RADIO);
		resnet101.setText("resnet 101");
		resnet101.setFont(fontArial);
		resnet50 = new Button(baseComponent, SWT.RADIO);
		resnet50.setText("resnet 50");
		resnet50.setFont(fontArial);

		if(resnetType == 0) {
			resnet50.setSelection(true);
			resnet101.setSelection(false);
		}

		//■GPU
		label_gpu = new Label(baseComponent, SWT.NONE);
		label_gpu.setText("Images per gpu ");
		label_gpu.setFont(fontArial);
		text_gpu_ubuntu = new Text(baseComponent,SWT.SINGLE|SWT.BORDER | SWT.CENTER);
		text_gpu_ubuntu.addVerifyListener(new VerifyListener() {
			public void verifyText(VerifyEvent ve) {
        	    if (ve.character < 0x0020) {
        	    } else if (ve.character >= 0x0030 && ve.character <= 0x0039) {
        	    } else ve.doit = false;//上記以外の文字の場合は入力拒否
			}
        });
		text_gpu_ubuntu.setText(Integer.toString(imagesPerGPU));
		text_gpu_ubuntu.setFont(fontArial);

		//■出力先フォルダ
		label_foldername = new Label(baseComponent, SWT.NONE);
		label_foldername.setText("Select a folder to save the mask image and weight file.");
		label_foldername.setFont(fontArial);

		text_saveFoldername = new Text(baseComponent,SWT.SINGLE|SWT.BORDER);
		text_saveFoldername.setFont(fontArial);
		text_saveFoldername.setBackground(colorNote);
		Button button_saveFoldername = new Button(baseComponent, SWT.NONE);
		button_saveFoldername.setText("-");
		button_saveFoldername.setForeground(color128);
		button_saveFoldername.setBackground(color064);
		button_saveFoldername.setFont(fontArial);

		//■初期重みファイル
		label_filename = new Label(baseComponent, SWT.NONE);
		label_filename.setText("※ Use existing Initial weight file ( *.h5 )");
		label_filename.setFont(fontArial);
		text_initWeightFilename = new Text(baseComponent,SWT.SINGLE|SWT.BORDER);
		text_initWeightFilename.setFont(fontArial);
		text_initWeightFilename.setBackground(color247);
		button_initWeightFilename = new Button(baseComponent, SWT.NONE);
		button_initWeightFilename.setText("-");
		button_initWeightFilename.setForeground(color128);
		button_initWeightFilename.setBackground(color064);
		button_initWeightFilename.setFont(fontArial);

		//■マスク画像フォルダ
		final Label label_foldername2 = new Label(baseComponent, SWT.NONE);
		label_foldername2.setText("※ Use existing mask Image folder ( mask_img_* )");
		label_foldername2.setFont(fontArial);
		text_maskFoldername = new Text(baseComponent,SWT.SINGLE|SWT.BORDER);
		text_maskFoldername.setFont(fontArial);
		text_maskFoldername.setBackground(color247);
		Button button_maskFoldername = new Button(baseComponent, SWT.NONE);
		button_maskFoldername.setText("-");
		button_maskFoldername.setForeground(color128);
		button_maskFoldername.setBackground(color064);
		button_maskFoldername.setFont(fontArial);

		button_saveFoldername.addSelectionListener(
			new SelectionAdapter(){
				public void widgetSelected(SelectionEvent e){
			        DirectoryDialog fileDlg = new DirectoryDialog(shell,SWT.NONE);
			        fileDlg.setFilterPath(text_saveFoldername.getText());
			        fileDlg.setText("Select Folder");
			        fileDlg.setMessage("Select Folder");
			        final String foldername = fileDlg.open();
					if(foldername != null && !foldername.isEmpty()){
						text_saveFoldername.setText(foldername);
					}
				}
			}
      	);

		button_initWeightFilename.addSelectionListener(
			new SelectionAdapter(){
				public void widgetSelected(SelectionEvent e){
					FileDialog fileDlg = new FileDialog(shell, SWT.OPEN);
			        fileDlg.setText("Select File");
			        fileDlg.setFilterPath(System.getProperty("user.dir") + "/");
			        
			        if(DeepLearningType == 1 || DeepLearningType == 3){
			        	String ext[] = {"*.h5"};
			        	fileDlg.setFilterExtensions(ext);
			        }else if(DeepLearningType == 0 || DeepLearningType == 2){
			        	String ext[] = {"*.pth"};
			        	fileDlg.setFilterExtensions(ext);
			        }
  					
			        final String filename = fileDlg.open();
					if(filename != null && !filename.isEmpty()){
						text_initWeightFilename.setText(filename);
					}
				}
			}
      	);

		button_maskFoldername.addSelectionListener(
			new SelectionAdapter(){
				public void widgetSelected(SelectionEvent e){
			        DirectoryDialog fileDlg = new DirectoryDialog(shell,SWT.NONE);
			        fileDlg.setFilterPath(text_maskFoldername.getText());
			        fileDlg.setText("Select Folder");
			        fileDlg.setMessage("Select Folder");
			        final String foldername = fileDlg.open();
					if(foldername != null && !foldername.isEmpty()){
						text_maskFoldername.setText(foldername);
					}
				}
			}
      	);


		if(largeWindow) {
			baseComponent.setBounds(0, 0, 520, 310);

			label_foldername.setBounds(20, 20, 400, 20);
			text_saveFoldername.setBounds(25, 42, 430, 26);
			button_saveFoldername.setBounds(460, 40, 40, 28);

			label_stepperepoch.setBounds(40, 100, 120, 26);
			text_stepperepoch_ubuntu.setBounds(160, 100, 80, 26);
			label_gpu.setBounds(40, 140, 120, 26);
			text_gpu_ubuntu.setBounds(160, 140, 80, 26);

			label_epoch1.setBounds(263, 100, 135, 26);
			text_epoch1_ubuntu.setBounds(400, 100, 80, 26);

			resnet50.setBounds(270, 140, 100, 26);
			resnet101.setBounds(375, 140, 100, 26);

			label_filename.setBounds(20, 190, 465, 20);
			text_initWeightFilename.setBounds(25, 212, 430, 26);
			button_initWeightFilename.setBounds(460, 211, 40, 28);

			label_foldername2.setBounds(20, 245, 465, 20);
			text_maskFoldername.setBounds(25, 267, 430, 26);
			button_maskFoldername.setBounds(460, 266, 40, 28);
			
		}else {
			baseComponent.setBounds(0,0,480, 310);

			label_foldername.setBounds(20, 19, 400, 21);
			text_saveFoldername.setBounds(25, 40, 390, 21);
			button_saveFoldername.setBounds(420, 38, 40, 24);

			label_stepperepoch.setBounds(40, 100, 100, 21);
			text_stepperepoch_ubuntu.setBounds(140, 100, 80, 21);
			label_gpu.setBounds(40, 140, 100, 21);
			text_gpu_ubuntu.setBounds(140, 140, 80, 21);

			text_epoch1_ubuntu.setBounds(370, 100, 80, 21);
			label_epoch1.setBounds(256, 100, 110, 21);

			resnet50.setBounds(265, 140, 100, 21);
			resnet101.setBounds(355, 140, 100, 21);

			label_filename.setBounds(20, 188, 365, 21);
			text_initWeightFilename.setBounds(25, 210, 390, 21);
			button_initWeightFilename.setBounds(420, 208, 40, 24);

			label_foldername2.setBounds(20, 243, 365, 21);
			text_maskFoldername.setBounds(25, 265, 390, 21);
			button_maskFoldername.setBounds(420, 263, 40, 24);

		}
		
		adjustGUI();
		return baseComponent;
	
	}

	void adjustGUI() {
		
		//Matterport
		label_stepperepoch.setText("Step per epoch ");
		text_stepperepoch_ubuntu.setText(Integer.toString(50));
		text_epoch1_ubuntu.setText(Integer.toString(400));
		label_epoch1.setText("Number of epochs");
		resnet101.setVisible(true);
		resnet50.setVisible(true);
		label_gpu.setText("Images per gpu ");
	   	label_gpu.setVisible(true);
	   	text_gpu_ubuntu.setText(Integer.toString(imagesPerGPU));
	   	text_gpu_ubuntu.setVisible(true);
	   	label_filename.setText("※ Use existing Initial weight file ( *.h5 )");
	   	label_filename.setEnabled(true);
		text_initWeightFilename.setEnabled(true);
		button_initWeightFilename.setEnabled(true);

		text_saveFoldername.setText("");
		text_initWeightFilename.setText("");
		text_maskFoldername.setText("");
		
    	//GUI調整 Detectron2 
		if(DeepLearningType == 0){
	    	label_stepperepoch.setText("Ims per batch");
	    	text_stepperepoch_ubuntu.setText(Integer.toString(2));
	    	text_epoch1_ubuntu.setText(Integer.toString(500));
	    	
	    	label_epoch1.setText("       Max iteration");
	    	
	    	label_gpu.setText("Num workers");
	    	text_gpu_ubuntu.setText(Integer.toString(2));
	    	
			resnet101.setVisible(false);
			resnet50.setVisible(false);
			
			label_filename.setText("※ Use existing Initial weight file ( *.pth )");
    	}
		
		//GUI調整 YOLACT 
    	else if(DeepLearningType == 2){
	    	label_stepperepoch.setText("Save Interval ");
	    	text_stepperepoch_ubuntu.setText(Integer.toString(500));
	    	label_epoch1.setText("Validation Epoch");
	    	text_epoch1_ubuntu.setText(Integer.toString(500));
	    	
	    	label_gpu.setText("Batch Size ");
			resnet101.setVisible(false);
			resnet50.setVisible(false);
			label_filename.setText("※ Use existing Initial weight file ( *.pth )");
    	}

    	//GUI調整 StarDist
    	else if(DeepLearningType == 3){
	    	text_stepperepoch_ubuntu.setText(Integer.toString(4));
	    	text_epoch1_ubuntu.setText(Integer.toString(500));
	    	
	    	label_gpu.setVisible(false);
	    	text_gpu_ubuntu.setVisible(false);
	    	
			resnet101.setVisible(false);
			resnet50.setVisible(false);
			
			label_filename.setEnabled(false);//.setText("※ Use existing Initial weight file");
			text_initWeightFilename.setEnabled(false);
			button_initWeightFilename.setEnabled(false);
    	}
    	//GUI調整 CellPose 
    	else if(DeepLearningType == 4){
	    	label_stepperepoch.setText("Batch size ");
	    	text_stepperepoch_ubuntu.setText(Integer.toString(4));
	    	text_epoch1_ubuntu.setText(Integer.toString(500));
	    	
	    	label_gpu.setVisible(false);
	    	text_gpu_ubuntu.setVisible(false);
	    	
			resnet101.setVisible(false);
			resnet50.setVisible(false);
			
			label_filename.setText("※ Use existing Initial weight file");
    	}
	}
	
	public String getWeightFilename(){

		setProperties();
		
		String filename = null;
		
		if(DeepLearningType == 0 || DeepLearningType == 1 || DeepLearningType == 2){
	
			MessageBox box = new MessageBox(shell, SWT.OK|SWT.CANCEL | SWT.ICON_INFORMATION);
			box.setText(" " + toString());
			
			String fileExt = "*.h5";
			if(DeepLearningType == 0 || DeepLearningType == 2) {
				fileExt = "*.pth";
			}
			
			box.setMessage("First, start the DL detection service. \n Select a weight file ("+ fileExt + ") in the next step.");
			int ret = box.open();
			switch(ret){
			    case SWT.OK:
			       break;
			    case SWT.CANCEL:
			    	return null;
			}
			FileDialog fd =	new FileDialog(shell, SWT.OPEN);
			fd.setText("Select Weight File");
			
			String ext[] = {fileExt};
			fd.setFilterExtensions(ext);

			filename = fd.open();

            //重みファイルがあるかチェック
			if(filename == null){
				return null;
			}

	      	if(Files.notExists(Paths.get(filename))){
				MessageBox box2 = new MessageBox(shell, SWT.OK | SWT.ICON_WARNING);
				box2.setText(" " + toString());
				box2.setMessage("Weight File dosen't exist.");
				int ret2 = box2.open();
				switch(ret2){
				    case SWT.OK:
				       break;
				}
				return null;
	      	}
	      	
		//StarDistの場合 フォルダー名を選択する
		}else if(DeepLearningType == 3){

			MessageBox box = new MessageBox(shell, SWT.OK|SWT.CANCEL | SWT.ICON_INFORMATION);
			box.setText(" " + toString());
			box.setMessage("First, start the DL detection service. \n Select a weight folder in the next step.");
			int ret = box.open();
			switch(ret){
			    case SWT.OK:
			       break;
			    case SWT.CANCEL:
			    	return null;
			}

	        DirectoryDialog fileDlg = new DirectoryDialog(shell,SWT.NONE);
	        //fileDlg.setFilterPath(text_maskFoldername.getText());
	        fileDlg.setText("Select Folder");
	        fileDlg.setMessage("Select Folder");
	        filename = fileDlg.open();

	        //if(!(filename.equals("2D_versatile_fluo") || filename.equals("2D_versatile_he") || filename.equals("2D_paper_dsb2018") || filename.equals("2D_demo"))) {
	        
		      	if(filename == null || Files.notExists(Paths.get(filename))){
					MessageBox box2 = new MessageBox(shell, SWT.OK | SWT.ICON_WARNING);
					box2.setText(" " + toString());
					box2.setMessage("Weight folder dosen't exist.");
					int ret2 = box2.open();
					switch(ret2){
					    case SWT.OK:
					       break;
					}
					return null;
		      	}
	
		      	//フォルダの中のh5ファイルを確認
		      	File file = new File(filename);
		        File files[] = file.listFiles();
		        boolean flg = false;
		        for (int i=0; i<files.length; i++) {
		        	String filePath = files[i].toString();
		        	if ((filePath.contains("weights_best.h5") || filePath.contains("weights_last.h5") || filePath.contains("weights_now.h5"))) {
		        		flg = true;
		        		break;
		        	}
		        }
				if (!flg) {
					MessageBox box2 = new MessageBox(shell, SWT.OK | SWT.ICON_WARNING);
					box2.setText(" " + toString());
					box2.setMessage("Weight file (*.h5) dosen't exist.");
					int ret2 = box2.open();
					switch(ret2){
					    case SWT.OK:
					       break;
					}
					return null;
				}
	        //}
				
		//CellPoseの場合
		}else if(DeepLearningType == 4){
			
				//weightファイルの拡張子は無し！
				MessageBox box = new MessageBox(shell, SWT.YES | SWT.NO | SWT.CANCEL | SWT.ICON_INFORMATION);
				box.setText(" " + toString());
				box.setMessage("Use your own weights file?");
				int ret = box.open();
				
				if(ret == SWT.YES){
  					FileDialog fd =	new FileDialog(shell, SWT.OPEN);
  					fd.setText("Select Weight File");
  					filename = fd.open();

		            //重みファイルがあるかチェック
  					if(filename == null){
  						return null;
  					}
  			      	if(Files.notExists(Paths.get(filename))){
  						MessageBox box2 = new MessageBox(shell, SWT.OK | SWT.ICON_WARNING);
  						box2.setText(" " + toString());
  						box2.setMessage("Weight File dosen't exist.");
  						int ret2 = box2.open();
  						switch(ret2){
  						    case SWT.OK:
  						       break;
  						}
  						return null;
  			      	}
				}else if(ret == SWT.NO){
					MessageBox box2 = new MessageBox(shell, SWT.YES | SWT.NO | SWT.CANCEL | SWT.ICON_INFORMATION);
					box2.setText(" " + toString());
					box2.setMessage("Is the target cell nucleus?");
					int ret2 = box2.open();
					if(ret2 == SWT.YES){
						filename = "nuclei";
					}else if(ret2 == SWT.NO){
						filename = "cyto";
					}else{
						return null;
					}
				}else{
					return null;
				}

		}
		return filename;
	}

	public String getDetectCommand(){
		String weightFilename = getWeightFilename();
		if(weightFilename == null) return null;

		//Detectron2
		if(DeepLearningType == 0){
			return DeepLearningService.get(0) + " --mode=detect --config_file=COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --weights=" + weightFilename;

		//Matterport	
		}else if(DeepLearningType == 1){
			return DeepLearningService.get(1)  +" detect --weights=" + weightFilename + " --numclass=2";

		//YOLACT++
		}else if(DeepLearningType == 2){
			return DeepLearningService.get(2) + " --mode=detect --trained_model=" + weightFilename + " --score_threshold=0.15 --top_k=500";

		//StarDist
		}else if(DeepLearningType == 3){
			return DeepLearningService.get(3) + " detect --pretrained_model " + weightFilename;		
			//reutnr DeepLearning_batchFile + " detect --pretrained_model nuclei --diameter 0. --use_gpu";
			
		//CellPose
		}else if(DeepLearningType == 4){
			String useGpu = " --use_gpu"; 
			if(cellposeCpuMode == 1) useGpu = " ";
			
			return DeepLearningService.get(4) + " detect --pretrained_model " + weightFilename + useGpu;		
			//reutnr DeepLearning_batchFile + " detect --pretrained_model nuclei --diameter 0. --use_gpu";
		}
		
		return "ERROR";
	}

	public String getTrainCommand(String initWeightFilename, String maskImageFoldername, String saveWeightFoldername){

		Display.getDefault().syncExec(new Runnable() {
			public void run() {
				stepperepoch = Integer.parseInt(text_stepperepoch_ubuntu.getText());
				epoch1 = Integer.parseInt(text_epoch1_ubuntu.getText());
				epoch0 = (int)(epoch1/2);
				gpu = Integer.parseInt(text_gpu_ubuntu.getText());
				resnet = 101;
		    	if(resnet50.getSelection()) {
		    		resnet = 50;
		    	}
		    	saveFoldername = text_saveFoldername.getText();
			}
		});

		//Detectron2
		if(DeepLearningType == 0){
			String jsonFilename = maskImageFoldername + "/annotations.json";
			//String datasetFoldername2 = "E:/CellTrackService/detectron2_RE/datasets/phase_contrast/train";
			String weightFilename = "default";
			if(!initWeightFilename.equals("default")) weightFilename = initWeightFilename;
			String outputFoldername = saveFoldername;
			int maxIter = epoch1;//100000;
			int numWorkers = gpu;//2;
			int imsPerBatch = stepperepoch;//2;
			float baseLR = (float) 0.00025;
			int batchSizePerImage = 512;
			return DeepLearningService.get(0) + " --mode=train --config_file=COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --dataset_name=phase_contrast_train --json_file=" + jsonFilename + " --image_root=" + maskImageFoldername + " --weights=" + weightFilename + " --output_dir=" + outputFoldername + " --max_iter=" + maxIter + " --num_workers=" + numWorkers + " --ims_per_batch=" + imsPerBatch + " --base_lr=" + baseLR + " --batch_size_per_image=" + batchSizePerImage;

		//Matterport
		}else if(DeepLearningType == 1){
			return DeepLearningService.get(1) + " train --dataset=" + maskImageFoldername + " --weights=" + initWeightFilename + " --logs=" + saveFoldername + " --stepsperepoch=" + stepperepoch + " --epoch0=" + epoch0 + " --epoch1=" + epoch1 + " --typeresnet=resnet" + resnet + " --gpu=" + gpu + " --numclass=2";

		//YOLACT++
		}else if(DeepLearningType == 2){
			int save_interval = stepperepoch;
			int validation_epoch = epoch1;
			
			String command = DeepLearningService.get(2) + " --mode=train  --config=yolact_plus_base_config --batch_size=" + gpu + " --save_interval=" + save_interval + " --validation_epoch=" + validation_epoch + " --dataset_folder=" + maskImageFoldername + "/ --numclass=2 --num_workers=0 --save_folder=" + saveWeightFoldername + "/ --weights=" + initWeightFilename;
			if(checkUnicode(command)) return "ERROR";
			return command;
			
		//StarDist
		}else if(DeepLearningType == 3){
			
	      	//教師画像フォルダ内のファイル数確認
	      	File file = new File(maskImageFoldername);
	        File files[] = file.listFiles();
	        if(files.length < 4) {
				Display.getDefault().syncExec(new Runnable() {
					public void run() {
						System.out.println("Error マスク画像は2枚以上必要です。");
						MessageBox box = new MessageBox(shell, SWT.OK | SWT.ICON_WARNING);
						box.setText("");
				        box.setMessage("Not enough training data. At least two images are required.");
						int ret = box.open();
						switch(ret){
						    case SWT.OK:
						       break;
						}
					}
				});
				return "ERROR";
	        }

			String command = DeepLearningService.get(3) + " train --dataset " + maskImageFoldername + "/ --model_save_folder " + saveWeightFoldername + " --stepperepoch " + stepperepoch + " --epochs " + epoch1;
			if(checkUnicode(command)) return "ERROR";
			return command;
			
		//CellPose
		}else if(DeepLearningType == 4){
			//String strWeight = "None"; //None を読み込む
			//String strWeight = "cyto";//c:\Users\ ser1\.cellpose\models\cytotorch_0 を読み込む
			String strWeight = "cyto2";//cyto2torch_0  を読み込む
			if(!initWeightFilename.equals("default")) strWeight = initWeightFilename;
			
			String useGpu = " --use_gpu"; 
			if(cellposeCpuMode == 1) useGpu = " ";
			String command = DeepLearningService.get(4) + " train --train --train_size --img_filter _img --dir " + maskImageFoldername + "/ --logs=" + saveFoldername + " --pretrained_model " + strWeight + " --batch_size " + stepperepoch + " --n_epochs " + epoch1 + useGpu;
			if(checkUnicode(command)) return "ERROR";
			return command;
		}

		return "ERROR";
	}

	public int getMaskImageType() {
		
		//Detectron2
		if(DeepLearningType == 0){
			return 0;//COCO JSONファイル形式（背景クラス（０）無し） 
		
		//Matterport
		}else if(DeepLearningType == 1){
			if(imageWidth <= limitSize_train && imageHeight <= limitSize_train){
				return 2;//matterport形式（画像分割なし）　
			}else{
				return 1;//matterport形式（画像分割あり）
			}
		
		//YOLACT_coco
		}else if(DeepLearningType == 2){
			return 3;//COCO JSONファイル形式（背景クラス（０）有り） 
		
		//StarDist
		}else if(DeepLearningType == 3){
			return 4; //Cellpose用、16bitマスク画像
		
		//CellPose
		}else if(DeepLearningType == 4){
			return 4; //Cellpose用、16bitマスク画像
		}

		return 0;
	}

	public String getSaveFoldername() {
		return text_saveFoldername.getText();
	}

	public String getInitWeightFilename() {
		String filename = text_initWeightFilename.getText();
		if(filename.equals("")) {
			return "default";
		}
		if(Files.notExists(Paths.get(filename))){
			MessageBox box = new MessageBox(shell, SWT.OK | SWT.ICON_WARNING);
			box.setText("");
	        box.setMessage("Initial weight filename is wrong.");
			int ret = box.open();
			switch(ret){
			    case SWT.OK:
			       break;
			}
			return null;

		}else {
			return filename;
		}
	}

	public String getMaskImageFoldername() {
		String foldername = text_maskFoldername.getText();
    	if(foldername.equals("")) {
    		return "";
    	}
        if(Files.notExists(Paths.get(foldername))){
			MessageBox box = new MessageBox(shell, SWT.OK | SWT.ICON_WARNING);
			box.setText("");
	        box.setMessage("MaskImage foldername is wrong.");
			int ret = box.open();
			switch(ret){
			    case SWT.OK:
			       break;
			}
			return null;
        } else {
        	return foldername;
        }
	}

	public void setRoiCount(int count) {
		//MainAppの中で、ROI数をカウントし、StepPerEpocの値を調整
		if(DeepLearningType == 1){ 
			count = (int)(0.023*count+0.5);
			if(count < 10) count = 10;
			text_stepperepoch_ubuntu.setText(Integer.toString(count));
		}
	}
	
	public void setImageSize(int imageWidth, int imageHeight) {
		this.imageWidth = imageWidth;
		this.imageHeight = imageHeight;
	}

	public int getLimitImageSize() {
		if(DeepLearningType == 0 || DeepLearningType==2 || DeepLearningType == 3 || DeepLearningType == 4){
			return 10000; //サイズ制限を設けない！
		}
		return limitsize_detect;
	}

	public boolean checkUnicode(String str) {
		boolean flg = false;
		for(int i = 0 ; i < str.length() ; i++) {
			char ch = str.charAt(i);
			Character.UnicodeBlock unicodeBlock = Character.UnicodeBlock.of(ch);
			if (Character.UnicodeBlock.HIRAGANA.equals(unicodeBlock)
				|| Character.UnicodeBlock.KATAKANA.equals(unicodeBlock)
				|| Character.UnicodeBlock.HALFWIDTH_AND_FULLWIDTH_FORMS.equals(unicodeBlock)
				|| Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS.equals(unicodeBlock)
				|| Character.UnicodeBlock.CJK_SYMBOLS_AND_PUNCTUATION.equals(unicodeBlock)	
					) {
				flg = true;
				break;
			}
		}
		if(flg) {
			Display.getDefault().syncExec(new Runnable() {
				public void run() {
					System.out.println("Unicode error パスを見直してください");
					MessageBox box = new MessageBox(shell, SWT.OK | SWT.ICON_WARNING);
					box.setText("");
			        box.setMessage("Unicode error.");
					int ret = box.open();
					switch(ret){
					    case SWT.OK:
					       break;
					}
				}
			});
			return true;
		}else {
			return false;
		}
		
	}

}
