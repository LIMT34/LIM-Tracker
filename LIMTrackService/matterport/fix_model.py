import shutil

file_name = "C:/LIMTrackService/matterport/mrcnn/model.py"
back_name = file_name + ".original"
shutil.copy(file_name, back_name)

with open(file_name) as f:
    data_lines = f.read()

data_lines = data_lines.replace("checkpoints = filter(lambda f: f.startswith(\"mask_rcnn\"), checkpoints)", "checkpoints = filter(lambda f: f.startswith(\"weight\"), checkpoints)")
data_lines = data_lines.replace("regex = r\".*[/\\\\][\\w-]+(\\d{4})(\\d{2})(\\d{2})T(\\d{2})(\\d{2})[/\\\\]mask\\_rcnn\\_[\\w-]+(\\d{4})\\.h5\"", "regex = r\".*[/\\\\][\\w-]+(\\d{4})(\\d{2})(\\d{2})(\\d{2})(\\d{2})[/\\\\]weight\\_(\\d{4})\\.h5\"")
data_lines = data_lines.replace("self.log_dir = os.path.join(self.model_dir, \"{}{:%Y%m%dT%H%M}\".format(", "self.log_dir = os.path.join(self.model_dir, \"weight_{:%Y%m%d%H%M}\".format(now))")
data_lines = data_lines.replace("self.config.NAME.lower(), now))", "")
data_lines = data_lines.replace("self.checkpoint_path = os.path.join(self.log_dir, \"mask_rcnn_{}_*epoch*.h5\".format(", "self.checkpoint_path = os.path.join(self.log_dir, \"weight_*epoch*.h5\")")
data_lines = data_lines.replace("self.config.NAME.lower()))", "")
data_lines = data_lines.replace("callbacks += custom_callbacks", "callbacks = custom_callbacks")
data_lines = data_lines.replace("max_queue_size=100,", "max_queue_size=2,")

with open(file_name, mode="w") as f:
    f.write(data_lines)



