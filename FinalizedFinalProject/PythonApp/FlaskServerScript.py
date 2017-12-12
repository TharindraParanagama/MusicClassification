from flask import Flask,render_template,request
import FeatureExtractor as FE
import Predictor as P

app = Flask(__name__)

@app.route('/')
def main():
	  return render_template('Web.html')

@app.route('/userGuide.html')
def userGuide():
	  return render_template('userGuide.html')

@app.route('/getPath', methods=['POST'])
def getPath():
      path=request.form['path1']
      print path
      x=FE.getSourcePath(path)
      print x
      return x
@app.route('/setPath', methods=['POST'])
def setPath():
      path=request.form['path2']
      print path
      x=P.setSourcePath(path)
      print x
      return x


if __name__=='__main__':
	  app.run()
