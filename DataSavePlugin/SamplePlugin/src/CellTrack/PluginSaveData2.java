package CellTrack;

import java.util.ArrayList;

public class PluginSaveData2 implements PluginSaveDataIF {

	public String toString(){
			return "Save Data Test Plugin2";
	}
	
	public void exec(
			ArrayList<ArrayList<RoiInfo>> roiLists0, 
			ArrayList<ArrayList<RoiInfo>> roiLists1, 
			ArrayList<ArrayList<RoiInfo>> roiLists2, 
			ArrayList<ArrayList<RoiInfo>> roiLists3, 
			ArrayList<ArrayList<RoiInfo>> roiLists4, 
			ArrayList<ArrayList<RoiInfo>> roiLists5,
			double PixelSize, 
			int PixelSizeUnit, //0:um, 1:mm
			String TimeInterval) {

		String unit = "um";
		if(PixelSizeUnit != 0) unit = "mm";
		System.out.println("Save Data Plugin: PixelSize=" + PixelSize + unit + ", TimeInterval=" + TimeInterval);

		//System.out.println("PluginSaveData2: roiLists0.size()->" + roiLists0.size());
		//System.out.println("PluginSaveData2: roiLists0.get(0).size()->" + roiLists0.get(0).size());
		
		if(roiLists0 != null && roiLists0.size() != 0) {
			for(int imgNo = 0; imgNo < roiLists0.size(); imgNo++) {
				for(int i = 0; i < roiLists0.get(imgNo).size(); i++) {
					dispRoiInfo(roiLists0.get(imgNo).get(i), 0);
					
					if(roiLists1 != null && roiLists1.size() != 0 && roiLists1.get(imgNo).size() != 0) {
						dispRoiInfo(roiLists1.get(imgNo).get(i), 1);
					}
					if(roiLists2 != null && roiLists2.size() != 0 && roiLists2.get(imgNo).size() != 0) {
						dispRoiInfo(roiLists2.get(imgNo).get(i), 2);
					}
					if(roiLists3 != null && roiLists3.size() != 0 && roiLists3.get(imgNo).size() != 0) {
						dispRoiInfo(roiLists3.get(imgNo).get(i), 3);
					}
					if(roiLists4 != null && roiLists4.size() != 0 && roiLists4.get(imgNo).size() != 0) {
						dispRoiInfo(roiLists4.get(imgNo).get(i), 4);
					}
					if(roiLists5 != null && roiLists5.size() != 0 && roiLists5.get(imgNo).size() != 0) {
						dispRoiInfo(roiLists5.get(imgNo).get(i), 5);
					}
				}
			}
		}

	}
    
	private void dispRoiInfo(RoiInfo roiInfo, int ch) {

		System.out.println("-------------------------------------------------------");
		System.out.println("imgNo: " + roiInfo.imgNo);
		System.out.println("cellID: " + roiInfo.roiID);	
		System.out.println("Ch: " + ch);
		System.out.println("-------------------------------------------------------");
		System.out.println("cellID_P: " + roiInfo.roiPID);
		System.out.println("cellPos: " + roiInfo.roiPos_rect[0] + ", " + roiInfo.roiPos_rect[1] + ", " + roiInfo.roiPos_rect[2] + ", " + roiInfo.roiPos_rect[3] 
				+ ", " + roiInfo.roiPos_rect[4] + ", " + roiInfo.roiPos_rect[5] + ", " + roiInfo.roiPos_rect[6] + ", " + roiInfo.roiPos_rect[7]);

		System.out.println("cellPos_inner: " + roiInfo.roiPos_cir[0] + ", " + roiInfo.roiPos_cir[1] + ", " + roiInfo.roiPos_cir[2] + ", " + roiInfo.roiPos_cir[3] 
				+ ", " + roiInfo.roiPos_cir[4] + ", " + roiInfo.roiPos_cir[5] + ", " + roiInfo.roiPos_cir[6] + ", " + roiInfo.roiPos_cir[7]);

		System.out.println("circleRatio: " + roiInfo.circleRatio);

		//矩形領域
		System.out.println("AveBrightness_rect: " + roiInfo.AveBrightness_rect);
		System.out.println("RoiArea_rect: " + roiInfo.RoiArea_rect);
		System.out.println("TotalBrightness_rect: " + roiInfo.TotalBrightness_rect);
		System.out.println("MaxValue_rect: " + roiInfo.MaxValue_rect);
		System.out.println("MinValue_rect: " + roiInfo.MinValue_rect);
		System.out.println("Variance_rect: " + roiInfo.Variance_rect);
		System.out.println("SD_rect: " + roiInfo.SD_rect);
		
		//円領域
		System.out.println("AveBrightness_cir: " + roiInfo.AveBrightness_cir);
		System.out.println("RoiArea_cir: " + roiInfo.RoiArea_cir);
		System.out.println("TotalBrightness_cir: " + roiInfo.TotalBrightness_cir);
		System.out.println("MaxValue_cir: " + roiInfo.MaxValue_cir);
		System.out.println("MinValue_cir: " + roiInfo.MinValue_cir);
		System.out.println("Variance_cir: " + roiInfo.Variance_cir);
		System.out.println("SD_cir: " + roiInfo.SD_cir);
		
		//任意形状領域
		System.out.println("AveBrightness_seg: " + roiInfo.AveBrightness_seg);
		System.out.println("SegArea: " + roiInfo.SegArea);
		System.out.println("TotalBrightness_seg: " + roiInfo.TotalBrightness_seg);
		System.out.println("MaxValue_seg: " + roiInfo.MaxValue_seg);
		System.out.println("MinValue_seg: " + roiInfo.MinValue_seg);
		System.out.println("Variance_seg: " + roiInfo.Variance_seg);
		System.out.println("SD_seg: " + roiInfo.SD_seg);
		
		System.out.println("Perimeter4: " + roiInfo.Perimeter4);
		System.out.println("Perimeter8: " + roiInfo.Perimeter8);
		System.out.println("Roundness: " + roiInfo.Roundness);
		System.out.println("MinX_seg: " + roiInfo.MinX_seg);
		System.out.println("MaxX_seg: " + roiInfo.MaxX_seg);
		System.out.println("MinY_seg: " + roiInfo.MinY_seg);
		System.out.println("MaxY_seg: " + roiInfo.MaxY_seg);
		System.out.println("CenterPosX_seg: " + roiInfo.CenterPosX_seg);
		System.out.println("CenterPosY_seg: " + roiInfo.CenterPosY_seg);
		System.out.println("CenterPosGravityX_seg: " + roiInfo.CenterPosGravityX_seg);
		System.out.println("CenterPosGravityY_seg: " + roiInfo.CenterPosGravityY_seg);
		System.out.println("FeretWidth: " + roiInfo.FeretWidth);
		System.out.println("FeretHeight: " + roiInfo.FeretHeight);
		System.out.println("FeretRatio: " + roiInfo.FeretRatio);
		System.out.println("Direction: " + roiInfo.Direction);
		
		System.out.println("cellShapeX: " + roiInfo.shapeX);
		System.out.println("cellShapeY: " + roiInfo.shapeY);	
		System.out.println("cellShapeW: " + roiInfo.shapeW);
		System.out.println("cellShapeH: " + roiInfo.shapeH);
		
		if(roiInfo.segShape != null) {
			System.out.println();
			for(int j = 0; j < roiInfo.shapeH; j++) {
				for(int i = 0; i < roiInfo.shapeW; i++) {
					if(roiInfo.segShape[i + j * roiInfo.shapeW]) {
						System.out.print("■");
					}else {
						System.out.print("□");
					}
				}
				System.out.println();
			}
			System.out.println();
		}
		
	}
}