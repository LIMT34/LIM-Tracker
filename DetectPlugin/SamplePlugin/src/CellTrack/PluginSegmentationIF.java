package CellTrack;

import org.eclipse.swt.widgets.Composite;

public interface PluginSegmentationIF {
	
	//�p�����[�^�ݒ�GUI�擾
    public Composite getParameterSettingPanel(Composite parent, boolean largeWindow);
    
	//Param0
	public float getParam0();
	public void setParam0(float val);
	public float getParam0_max();
	public void setParam0_max(float val);
	
	//Param1
	public float getParam1();
	public void setParam1(float val);
	public float getParam1_max();
	public void setParam1_max(float val);
	
	//Param2
	public float getParam2();
	public void setParam2(float val);
	public float getParam2_max();
	public void setParam2_max(float val);
	
	//Param3
	public float getParam3();
	public void setParam3(float val);
	public float getParam3_max();
	public void setParam3_max(float val);
	
	//Param4
	public float getParam4();
	public void setParam4(float val);
	public float getParam4_max();
	public void setParam4_max(float val);
	
	//Param5
	public float getParam5();
	public void setParam5(float val);
	public float getParam5_max();
	public void setParam5_max(float val);
	
	//Param6
	public float getParam6();
	public void setParam6(float val);
	public float getParam6_max();
	public void setParam6_max(float val);
	
	//Param7
	public float getParam7();
	public void setParam7(float val);
	public float getParam7_max();
	public void setParam7_max(float val);
	
	//Param8
	public float getParam8();
	public void setParam8(float val);
	public float getParam8_max();
	public void setParam8_max(float val);
	
	//Param9
	public float getParam9();
	public void setParam9(float val);
	public float getParam9_max();
	public void setParam9_max(float val);

	//ROI�T�C�Y�ݒ�i�P�_���o���ɕK�v�j
	public int getRoiSize();
	public void setRoiSize(int val);
	
	//�f�t�H���g�P�x臒l�ݒ�
	public void setDefaultThreshold(int min, int max, int threshold);

	//GUI�A�N�e�B�u�ݒ�
	public void setEnabled(boolean flg);

    //�Z�O�����e�[�V����or�P�_���o�t���O�ݒ�
	public void setSegmentataionMode(boolean flg);
	
	//�������s
    public void exec(int[] inImg, int[] segImg, int width, int height);  
    
}