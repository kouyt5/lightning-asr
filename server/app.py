from flask import Flask, request
import sys
sys.path.append('../')
from predict import AsrTranslator
import io


app = Flask(__name__)
model_path = "/data/chenc/asr/lightning-asr/outputs/asr13x1-pad32/2021-08-19/16-39-05/checkpoints/last.ckpt"
asr_translator = AsrTranslator(model_path=model_path, map_location="cuda:0", lang="en")


@app.route("/", methods=["POST"])
def translate():
    file = request.files['audio']
    file_bin = io.BytesIO(file.read())
    return asr_translator.translate(file_bin)

app.debug = True
app.run()

# POST localhost:5000
#   formdata: audio="xxxx.wav"