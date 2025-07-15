from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os


PIPE_PATH = "pipe.pkl"
DF_PATH   = "df.pkl"

if not (os.path.exists(PIPE_PATH) and os.path.exists(DF_PATH)):
    raise FileNotFoundError("Make sure pipe.pkl and df.pkl sit next to app.py")

pipe = pickle.load(open(PIPE_PATH, "rb"))
df   = pickle.load(open(DF_PATH,  "rb"))


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        companies   = sorted(df["Company"].unique()),
        types       = sorted(df["TypeName"].unique()),
        cpu_brands  = sorted(df["Cpu brand"].unique()),
        gpu_brands  = sorted(df["Gpu brand"].unique()),
        os_options  = sorted(df["os"].unique()),
        rams        = [2,4,6,8,12,16,24,32,64],
        hdds        = [0,128,256,512,1024,2048],
        ssds        = [0,8,128,256,512,1024],
        resolutions = [
            "1920x1080","1366x768","1600x900","3840x2160",
            "3200x1800","2880x1800","2560x1600","2560x1440","2304x1440"
        ],
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        f = request.form  

       
        company     = f["company"]
        type_name   = f["type"]
        ram         = int(f["ram"])
        weight      = float(f["weight"])
        touchscreen = 1 if f["touchscreen"] == "Yes" else 0
        ips         = 1 if f["ips"] == "Yes" else 0
        screen_size = float(f["screen_size"])
        res         = f["resolution"]
        cpu         = f["cpu"]
        hdd         = int(f["hdd"])
        ssd         = int(f["ssd"])
        gpu         = f["gpu"]
        os_name     = f["os"]

        
        X_res, Y_res = map(int, res.split("x"))
        ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

        
        query = pd.DataFrame(
            [[company, type_name, ram, weight, touchscreen, ips, ppi,
              cpu, hdd, ssd, gpu, os_name]],
            columns=["Company","TypeName","Ram","Weight","Touchscreen","IPS","ppi",
                     "Cpu brand","HDD","SSD","Gpu brand","os"]
        )

        log_price = pipe.predict(query)[0]
        price = int(np.exp(log_price))         

        return render_template("result.html", price=price)

    except Exception as e:
        
        return render_template("result.html", price=None, error=str(e)), 500



if __name__ == "__main__":
    app.run(debug=True, port=5000)
