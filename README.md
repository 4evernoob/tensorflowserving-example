# proyectF

 Docker + tensorflow test this repo is a test of  tensorflow/serving serving a neural network  using a docker container. I use jenkins to deploy it into heroku.
 working app in 
 https://eltestotenso.herokuapp.com/v1/models/winetest:predict
 
 payload is 
 {
 
"instances":[[6.796e-02 ,3.506e-03, 3.135e-03, 2.636e-02, 8.340e-01, 2.309e-03, 12.405e-04,
  8.978e-04, 1.340e-03, 1.012e-02, 1.040e-03, 2.252e-03, 10.904e-01]]
	
}
 

I used wine dataset as a test.
Next is deployment of image classifier
