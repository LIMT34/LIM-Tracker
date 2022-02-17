package CellTrack;

import java.util.ArrayList;

public interface PluginSaveDataIF {

	public void exec(
			ArrayList<ArrayList<RoiInfo>> roiLists0, 
			ArrayList<ArrayList<RoiInfo>> roiLists1, 
			ArrayList<ArrayList<RoiInfo>> roiLists2, 
			ArrayList<ArrayList<RoiInfo>> roiLists3, 
			ArrayList<ArrayList<RoiInfo>> roiLists4, 
			ArrayList<ArrayList<RoiInfo>> roiLists5,
			double PixelSize, 
			int PixelSizeUnit, //0:um, 1:mm
			String TimeInterval);  //00:00:00
 
}