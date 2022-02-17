package CellTrack;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.MouseEvent;
import org.eclipse.swt.events.MouseListener;
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
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Label;
import org.eclipse.swt.widgets.Scale;
import org.eclipse.swt.widgets.Text;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class PluginSegmentation2 implements PluginSegmentationIF {
	private int ubuntu_text_h = 21;//20;//win:20, ubuntu:21

	public String toString(){
	     return "OpenCV Segmentation Plugin";
	}
    
    public static void main(String[] args){}
    
    public PluginSegmentation2(){}
    
	private boolean flgSegmentation = false;
	private int threshold = 10000;
	private int cellRad = 8;

	public Color color064 = new Color(Display.getDefault(), 235,235,235);
	public Color color128 = new Color(Display.getDefault(), 110,110,110);
	public Font fontArial8;

	Composite DetectProcPanelPlugin;
	Label label_Threshold;
	Scale slider_Threshold;
	Text text_Threshold;

	Label label_CellSize;
	Scale slider_CellSize;
	Text text_CellSize;

	Label label_ROISize;
	Text text_ROISize;
	
	Label label_Dilation;
	Button button_Dilation;
	
	int ThresholdMin = 1;
	int ThresholdMax = 65535;
	int ThresholdTmp = 10000;

	int ROISizeIni = 15;
	int RoiSizeMin = 8;
	int CellSizeMin = 1;
	int CellSizeMax = 50;

	Label dummyLabel1;
	Label dummyLabel2;
	Label dummyLabel3;
	Label dummyLabel4;

	public void setDefaultThreshold(int min, int max, int threshold){
		ThresholdMin = min;
		ThresholdMax = max;
		ThresholdTmp = threshold;
		setParam0(ThresholdTmp);
		setParam0_max(ThresholdMax);
	}

	public void setSegmentataionMode(boolean flg){
		label_ROISize.setVisible(!flg);
		text_ROISize.setVisible(!flg);
		button_Dilation.setVisible(flg);
		label_Dilation.setVisible(flg);
		flgSegmentation = flg;
	}
	
	//Param0
	public float getParam0(){
		try {
			return (float)Integer.parseInt(text_Threshold.getText());
		} catch (NumberFormatException e) {
		    System.out.println("PluginSegmentation Threshold parse error: " + e);
		}
		return 255;
	}
	public void setParam0(float val){
		slider_Threshold.setSelection((int)val);
		text_Threshold.setText(Integer.toString((int)val));
	}
	public float getParam0_max(){
		return (float)slider_Threshold.getMaximum();
	}
	public void setParam0_max(float val){
		slider_Threshold.setMaximum((int)val);
	}
	
	//Param1
	public float getParam1(){
		try {
			return Integer.parseInt(text_CellSize.getText());
		} catch (NumberFormatException e) {
		    System.out.println("PluginSegmentation CellSize parse error: " + e);
		}
		return 15;
	}
	public void setParam1(float size){
		slider_CellSize.setSelection((int)size);
		text_CellSize.setText(Integer.toString((int)size));
	}
	public float getParam1_max(){
		return slider_CellSize.getMaximum();
	}
	public void setParam1_max(float size){
		slider_CellSize.setMaximum((int)size);
	}
	
	//Param2
	public float getParam2(){
		if(button_Dilation.getSelection() == true) {
			return (float) 1.5;
		}else {
			return (float) -1.0;
		}
	}
	public void setParam2(float coef){
		if(coef == -1.0) {
			button_Dilation.setSelection(false);
		}else {
			button_Dilation.setSelection(true);
		}
	}
	public float getParam2_max(){return 0;}
	public void setParam2_max(float size){}
	
	//Param3
	public float getParam3(){return 0;}
	public void setParam3(float size){}
	public float getParam3_max(){return 0;}
	public void setParam3_max(float size){}
	
	//Param4
	public float getParam4(){return 0;}
	public void setParam4(float size){}
	public float getParam4_max(){return 0;}
	public void setParam4_max(float size){}
	
	//Param5
	public float getParam5(){return 0;}
	public void setParam5(float size){}
	public float getParam5_max(){return 0;}
	public void setParam5_max(float size){}
	
	//Param6
	public float getParam6(){return 0;}
	public void setParam6(float size){}
	public float getParam6_max(){return 0;}
	public void setParam6_max(float size){}
	
	//Param7
	public float getParam7(){return 0;}
	public void setParam7(float size){}
	public float getParam7_max(){return 0;}
	public void setParam7_max(float size){}
	
	//Param8
	public float getParam8(){return 0;}
	public void setParam8(float size){}
	public float getParam8_max(){return 0;}
	public void setParam8_max(float size){}
	
	//Param9
	public float getParam9(){return 0;}
	public void setParam9(float size){}
	public float getParam9_max(){return 0;}
	public void setParam9_max(float size){}

	public int getRoiSize(){
		try {
			return Integer.parseInt(text_ROISize.getText());
		} catch (NumberFormatException e) {
		    System.out.println("PluginSegmentation RoiSize parse error: " + e);
		}
		return 15;
	}
	
	public void setRoiSize(int size){
		text_ROISize.setText(Integer.toString(size));
	}
	
	public void setEnabled(boolean flg){
		slider_Threshold.setEnabled(flg);
		text_Threshold.setEnabled(flg);
		slider_CellSize.setEnabled(flg);
		text_CellSize.setEnabled(flg);
		text_ROISize.setEnabled(flg);
	}
	
	public void exec(final int[] inImg, final int[] segImg, final int width, final int height) {
		Display.getDefault().syncExec(new Runnable() {
			public void run() {
				if(flgSegmentation){
					opencvDetection(inImg, segImg, width, height, threshold, cellRad, true);
				}else {
					opencvDetection(inImg, segImg, width, height, threshold, cellRad, false);
				}
			}
		});
	}

	public Composite getParameterSettingPanel(Composite parent, boolean largeWindow){
		
		//CellTrackGUI直下にopencv_java430.dllをコピー
		//Plugin直下にopencv-430.jarをコピー
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("OpenCV loadLibrary " + Core.NATIVE_LIBRARY_NAME);
        
  		//----------------------------------------------------------------------------------
  		//フォントサイズ調整
		int fontSizeMinimize = 0;
  		int limitSize = 64;
  		if(largeWindow){
  			limitSize = 85;
  		}
  		int minimize = 0; 
  		Label label_dammy = new Label(parent.getShell(), SWT.NONE);
  		for(int i = -5; i <= 5; i++){
	  		label_dammy.setFont(new Font(Display.getDefault(), "Arial", 8+i, SWT.NONE));
	  		Point textSize = new GC(label_dammy).textExtent("-------------");
	  		if(textSize.x <= limitSize){
	  			minimize = i;
	  		}else{
	  			break;
	  		}
  		}
  		fontSizeMinimize += (-minimize);
  		//----------------------------------------------------------------------------------
		
  		fontArial8 = new Font(Display.getDefault(), "Arial", 8-fontSizeMinimize, SWT.NONE);

		DetectProcPanelPlugin = new Composite(parent, SWT.NONE);
		DetectProcPanelPlugin.setBackground(color064);
		DetectProcPanelPlugin.setBounds(16, 31-5, 510, 90-18);

		button_Dilation = new Button(DetectProcPanelPlugin, SWT.CHECK);
		
		text_Threshold = new Text(DetectProcPanelPlugin,SWT.SINGLE|SWT.BORDER | SWT.CENTER);
		text_CellSize = new Text(DetectProcPanelPlugin,SWT.SINGLE|SWT.BORDER | SWT.CENTER);
		text_ROISize = new Text(DetectProcPanelPlugin,SWT.SINGLE|SWT.BORDER | SWT.CENTER);
		dummyLabel1 = new Label(DetectProcPanelPlugin, SWT.NONE);
		dummyLabel2 = new Label(DetectProcPanelPlugin, SWT.NONE);
		dummyLabel3 = new Label(DetectProcPanelPlugin, SWT.NONE);
		 
		dummyLabel1.setBackground(color064);
		dummyLabel2.setBackground(color064);
		dummyLabel3.setBackground(color064);
		
		 //認識4段目
		 label_CellSize = new Label(DetectProcPanelPlugin, SWT.NONE);
		 label_CellSize.setText("Cell Size");
		 label_CellSize.setBackground(color064);
		 label_CellSize.setFont(fontArial8);

		 text_CellSize.setText(Integer.toString(ROISizeIni));
		 text_CellSize.setFont(fontArial8);

		 
		 text_CellSize.addVerifyListener(new VerifyListener() {
				public void verifyText(VerifyEvent ve) {
	        	    if (ve.character < 0x0020) {
	        	    } else if (ve.character >= 0x0030 && ve.character <= 0x0039) {
	        	    } else ve.doit = false;//上記以外の文字の場合は入力拒否
				}
	        });

		 label_Threshold = new Label(DetectProcPanelPlugin, SWT.NONE);
		 label_Threshold.setText("Threshold");
		 label_Threshold.setBackground(color064);
		 label_Threshold.setFont(fontArial8);

		 dummyLabel4 = new Label(DetectProcPanelPlugin, SWT.NONE);
		 dummyLabel4.setBackground(color064);
		 
		 text_Threshold.setText(Integer.toString(ThresholdTmp));
		 text_Threshold.setFont(fontArial8);

		 text_Threshold.addVerifyListener(new VerifyListener() {
				public void verifyText(VerifyEvent ve) {
	        	    if (ve.character < 0x0020) {
	        	    } else if (ve.character >= 0x0030 && ve.character <= 0x0039) {
	        	    } else ve.doit = false;//上記以外の文字の場合は入力拒否
				}
	        });
		 
		 slider_Threshold = new Scale(DetectProcPanelPlugin, SWT.NONE);
		 slider_Threshold.setMinimum(ThresholdMin);
		 slider_Threshold.setMaximum(ThresholdMax);
		 slider_Threshold.setSelection(ThresholdTmp);
		 slider_Threshold.setIncrement(1);
		 slider_Threshold.setBackground(color064);
		 slider_Threshold.addSelectionListener(new SelectionAdapter() {
			public void widgetSelected(SelectionEvent e) {
		         slider_Threshold.setToolTipText("Th: " + slider_Threshold.getSelection());
		         text_Threshold.setText(Integer.toString(slider_Threshold.getSelection()));
		     }
		 });

		 slider_CellSize = new Scale(DetectProcPanelPlugin, SWT.NONE);
		 slider_CellSize.setMinimum(CellSizeMin);
		 slider_CellSize.setMaximum(CellSizeMax);
		 slider_CellSize.setSelection(ROISizeIni);
		 slider_CellSize.setIncrement(1);
		 slider_CellSize.setBackground(color064);
		 slider_CellSize.addSelectionListener(new SelectionAdapter() {
			public void widgetSelected(SelectionEvent e) {
		         slider_CellSize.setToolTipText("CellSize: " + slider_CellSize.getSelection());
		         text_CellSize.setText(Integer.toString(slider_CellSize.getSelection()));
		     }
		 });

		 label_Threshold.setBounds(0, 15, 56, ubuntu_text_h);
		 slider_Threshold.setBounds(53, 2, 268, 32);//■
		 text_Threshold.setBounds(319, 12, 56, ubuntu_text_h);

		 dummyLabel1.setBounds(0, 0, 380, 11);
		 dummyLabel2.setBounds(0, 33, 380, 11);
		 dummyLabel3.setBounds(0, 66, 380, 11);

		 label_CellSize.setBounds(0, 48, 50, ubuntu_text_h);
		 slider_CellSize.setBounds(53, 35, 195, 32);//■
		 text_CellSize.setBounds(244, 45, 30, ubuntu_text_h);

	     //Ubuntu
		 String os = System.getProperty("os.name");
	     if(!(os.equals("Windows 7") || os.equals("Windows 10"))) {
	    	 slider_Threshold.setBounds(61, 2, 252, 42);//■
	    	 slider_CellSize.setBounds(61, 35, 179, 42);//■
	     }

		 label_CellSize.setForeground(color128);
		 label_Threshold.setForeground(color128);

		 label_ROISize = new Label(DetectProcPanelPlugin, SWT.NONE);

		 label_ROISize.setFont(fontArial8);
		 label_ROISize.setForeground(color128);
		 label_ROISize.setBackground(color064);
		 label_ROISize.setText("ROI Size");

		 text_ROISize.setText(Integer.toString(ROISizeIni));
		 text_ROISize.setFont(fontArial8);

		 label_ROISize.setBounds(293, 48, 50, ubuntu_text_h);
		 text_ROISize.setBounds(345, 45, 30, ubuntu_text_h);

		 label_Dilation = new Label(DetectProcPanelPlugin, SWT.NONE);

		 label_Dilation.setFont(fontArial8);
		 label_Dilation.setForeground(color128);
		 label_Dilation.setBackground(color064);
		 label_Dilation.setText("Dilation");

		 button_Dilation.setVisible(false);
		 label_Dilation.setVisible(false);

		 label_Dilation.addMouseListener(new MouseListener(){
			    public void mouseDoubleClick(MouseEvent e){}
				public void mouseDown(MouseEvent e) {
					if(button_Dilation.getSelection()){
						button_Dilation.setSelection(false);
					}else{
						button_Dilation.setSelection(true);
					}
				}
				public void mouseUp(MouseEvent e) {}
			});
		 
		 button_Dilation.setBounds(304, 45, ubuntu_text_h-1, ubuntu_text_h);
		 label_Dilation.setBounds(326, 48, 50, ubuntu_text_h);

		 if(largeWindow) {
		 	DetectProcPanelPlugin.setBounds(16, 31, 510, 90);

		 	label_Threshold.setBounds(0, 17, 76, ubuntu_text_h);
		 	slider_Threshold.setBounds(73, 4, 363, 32);//■
		 	text_Threshold.setBounds(435, 14, 70, 24);

		 	dummyLabel1.setBounds(0, -5, 505, 18);
		 	dummyLabel2.setBounds(0, 35, 505, 18);
		 	dummyLabel3.setBounds(0, 75, 505, 11);
		 	dummyLabel4.setBounds(62, 6, 14, 60);//■

		 	label_CellSize.setBounds(0, 57, 76, ubuntu_text_h);
		 	slider_CellSize.setBounds( 73, 44, 270, 32);//■
		 	text_CellSize.setBounds(342, 54, 40, 24);

		 	label_ROISize.setBounds(395, 57, 70, ubuntu_text_h);
		 	text_ROISize.setBounds(465, 54, 40, 24);
		 	
		 	button_Dilation.setBounds(422, 56, ubuntu_text_h-1, ubuntu_text_h);
		 	label_Dilation.setBounds(442, 57, 70, ubuntu_text_h);

		 	if(System.getProperty("os.name").equals("Windows 7") ){
		 		dummyLabel1.setBounds(0, 0, 505, 13);
		 		slider_Threshold.setBounds(73, 3, 363, 37);//■
		 		dummyLabel2.setBounds(0, 40, 505, 13);
		 		slider_CellSize.setBounds(73, 43, 270, 37);//■
		 		dummyLabel3.setBounds(0, 80, 505, 6);
		 		dummyLabel4.setBounds(62, 6,  14, 60);//■
		 	}
		 }
		 
		 setEnabled(false);
		 return DetectProcPanelPlugin;
	 }

	//スポット検出⇒SegImgにラベルリング画像を入れて返す。
	private boolean opencvDetection(int[] inImg, int[] segImg, int width, int height, int threshold, int cellSize, boolean flgSegmentation){

		System.out.println("PluginSegmentation cellDetect");
		System.out.println(" Seg Param: Th" + threshold + ", Rad" + cellSize );

		Mat inImgCv = Mat.zeros(height, width, CvType.CV_8UC1);
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				inImgCv.put(j, i, (inImg[i + j * width] >> 8));  //16->8bit
			}
		}

        Imgproc.threshold(inImgCv, inImgCv, (threshold>>8), 255, Imgproc.THRESH_BINARY);// | Imgproc.THRESH_OTSU);
		
        Mat labelImg = new Mat(height, width, CvType.CV_32S);
        Mat stats = new Mat();
        Mat cents = new Mat();
        int nLabels = Imgproc.connectedComponentsWithStats(inImgCv, labelImg, stats, cents, 8, CvType.CV_32S);

        if(flgSegmentation) {
	        //Segmentation時
			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					double[] tmp = labelImg.get(j, i);
					segImg[i + j * width] = (int) tmp[0];
				}
			}
        }else {
            //Detection時
            int segNo = 1;
            double[] centInfo = new double[2];
            for(int i = 1; i < stats.rows(); i++) {
            	cents.row(i).get(0, 0, centInfo);
            	segImg[(int)centInfo[0] + (int)centInfo[1] * width] = segNo;
            	//System.out.println("pos no:" + segNo + ", xy:"  + centroidInfo[0] + ", " + centroidInfo[1]);
            	segNo++;
            	
            }
        }
        
        stats.release();
        cents.release();
		return true;
	}
}