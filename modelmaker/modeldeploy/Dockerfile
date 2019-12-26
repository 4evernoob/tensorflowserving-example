FROM ubuntu:18.04
RUN apt-get update && apt-get install -y --no-install-recommends nginx curl gnupg2 wget
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list
RUN wget --no-check-certificate https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg 
RUN apt-key add tensorflow-serving.release.pub.gpg
RUN apt-get update && apt-get install -y tensorflow-model-server sed gettext-base


# Copy our model folder to the container
COPY eltesto /eltesto
# run rest api
CMD tensorflow_model_server --port=$PORT \
 --model_name=winetest \
 --model_base_path=/eltesto

