package CellTrack;

import java.util.ArrayList;

import org.eclipse.swt.widgets.Composite;

public interface PluginDeepLearningIF {

	//学習ボタン押下後に表示する学習パラメータ設定パネルを返す。（windowSize: 1=ノーマル、2=拡大）
	public Composite getParameterSettingPanel(Composite parent, boolean largeWindow);

	//DLコマンド（認識）を返す。
	public String getDetectCommand();
	
	//DLコマンド（学習）を返す。
	public String getTrainCommand(String initWeightFilename, String maskImageFoldername, String saveWeightFoldername);

	//マスク画像のタイプを返す。
	public int getMaskImageType();
	
	//保存用フォルダ名を返す。
	public String getSaveFoldername();

	//初期重みファイル名を返す。
	public String getInitWeightFilename();
  
	//マスク画像保存用フォルダ名を返す。
	public String getMaskImageFoldername();
    
	//現時点で存在するROIの総数を設定。
	public void setRoiCount(int count);
	
	//処理を行う画像サイズを設定。
	public void setImageSize(int imageWidth, int imageHeight);
	
	//認識時の画像最大サイズ(当該サイズ以上の画像は分割して処理される)。	
	public int getLimitImageSize();
	
	public ArrayList<String> getDeepLearningName();
	
	public void setDeepLearningName(String deepLearningName);

}