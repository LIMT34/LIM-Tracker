
package CellTrack;
import java.io.Serializable;

@SuppressWarnings("serial")
public class RoiInfo implements Serializable  {
	public int roiID;
	public int roiPID;
	
	public int maskLabelNo = 0;
	
	public int roiID_LAP = -1;
	public int roiPID_LAP = 0;
	
	public int splitNum = 0;
	public int lineageID = 0;
	
	public float circleRatio = (float) 1.0;
	
	public int imgNo;
	public float lineagePos;
	public float[] roiPos_rect;
	public float[] roiPos_cir;
	
	//左上X,Y
	//cellPos[4]
	//cellPos[0]
	//左下X,Y
	//cellPos[5]  
	//cellPos[1]
	//右上X,Y
	//cellPos[6]
	//cellPos[2]
	//右下X,Y
	//cellPos[7]
	//cellPos[3]

	//領域エリア形状情報
	public int shapeX = 0;
	public int shapeY = 0;	
	public int shapeW = 0;
	public int shapeH = 0;	
	public boolean[] segShape;
    
	public int seqNo = 0;
	public int isMoving;
	public int isScaling;	
	public int isSelected;	
	public boolean isLocking = false;

	//矩形領域
	public double AveBrightness_rect;
	public int RoiArea_rect;
	public int TotalBrightness_rect;
	public int MaxValue_rect;
	public int MinValue_rect;
	public int MedValue_rect;
	public double Variance_rect;
	public double SD_rect;
	//円領域
	public double AveBrightness_cir;
	public int RoiArea_cir;
	public int TotalBrightness_cir;
	public int MaxValue_cir;
	public int MinValue_cir;
	public int MedValue_cir;
	public double Variance_cir;
	public double SD_cir;
	//任意形状領域
	public double AveBrightness_seg;
	public int SegArea;
	public double Perimeter4;
	public double Perimeter8;
	public double Roundness;
	public int MinX_seg;
	public int MinY_seg;
	public int MaxX_seg;
	public int MaxY_seg;
	public double CenterPosX_seg;
	public double CenterPosY_seg;
	public int FeretWidth;
	public int FeretHeight;
	public double FeretRatio;
	public int SumX;
	public int SumY;
	public int SumX2;
	public int SumY2;
	public int SumXY;
	public double CenterPosGravityX_seg;
	public double CenterPosGravityY_seg;
	public double Direction;
	public int TotalBrightness_seg;
	public int MaxValue_seg;
	public int MinValue_seg;
	public int MedValue_seg;
	public double Variance_seg;
	public double SD_seg;
	public double AveBoundary;
	public int PixBound = -1;

	//コンストラクタ
    public RoiInfo(){
    	roiID = -1;
    	roiPID = 0;
    	imgNo = -1;
        lineagePos = -1.0f;
        roiPos_rect = new float[10];
        roiPos_cir = new float[10];

        roiPos_rect[8] = 0;
        roiPos_rect[9] = 0;
        roiPos_cir[8] = 0;
        roiPos_cir[9] = 0;

        isSelected = 0;

    	AveBrightness_rect = 0.0;
    	RoiArea_rect = 0;
    	TotalBrightness_rect = 0;
    	MaxValue_rect = 0;
    	MinValue_rect = 65535;
    	MedValue_rect = 0;
    	Variance_rect = 0.0;
    	SD_rect = 0.0;       
        
    	AveBrightness_seg = 0.0;
		SegArea = 0;
		TotalBrightness_seg = 0;
		MaxValue_seg = 0;
		MinValue_seg = 0;
		MedValue_seg = 0;
		Variance_seg = 0.0;
		SD_seg = 0.0;
		AveBoundary = 0.0;
		PixBound = 0;
		
		Perimeter4 = 0.0;
		Perimeter8 = 0.0;
		Roundness = 0.0;
		MinX_seg = 0;
		MinY_seg = 0;
		MaxX_seg = 0;
		MaxY_seg = 0;
		CenterPosX_seg = 0.0;
		CenterPosY_seg = 0.0;
		FeretWidth = 0;
		FeretHeight = 0;
		FeretRatio = 0.0;
		SumX = 0;
		SumY = 0;
		SumX2 = 0;
		SumY2 = 0;
		SumXY = 0;
		CenterPosGravityX_seg = 0.0;
		CenterPosGravityY_seg = 0.0;
		Direction = 0.0;
		
    	AveBrightness_cir = 0.0;
    	RoiArea_cir = 0;
    	TotalBrightness_cir = 0;
    	MaxValue_cir = 0;
    	MinValue_cir = 65535;
    	MedValue_cir = 0;
    	Variance_cir = 0.0;
    	SD_cir = 0.0;     
    }

	public int getCenterPosX() { return (int)((roiPos_rect[4] + roiPos_rect[5] + roiPos_rect[6] + roiPos_rect[7])/4.0 + 0.5);}
	public int getCenterPosY() { return (int)((roiPos_rect[0] + roiPos_rect[1] + roiPos_rect[2] + roiPos_rect[3])/4.0 + 0.5);}
}

