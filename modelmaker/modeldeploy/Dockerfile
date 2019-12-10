# Using the official tensorflow serving image from docker hub as base image
FROM ubuntu:18.04

# Installing NGINX
RUN apt-get update && apt-get install -y --no-install-recommends nginx curl gnupg2 wget
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list
RUN wget --no-check-certificate https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg 
RUN apt-key add tensorflow-serving.release.pub.gpg
RUN apt-get update && apt-get install -y tensorflow-model-server-universal sed
#RUN apt-get upgrade tensorflow-model-server-universal

# Copy our model folder to the container
COPY eltesto /eltesto

# Copy NGINX configuration to the container
COPY nginx.conf /etc/nginx/nginx.conf
RUN echo "daemon off;" >> /etc/nginx/nginx.conf
RUN sed -i -e 's/$PORT/'"$PORT"'/g' /etc/nginx/nginx.conf
# starts NGINX and TF serving pointing to our model
ENTRYPOINT (/usr/sbin/nginx)| (tensorflow_model_server --rest_api_port=15000 \
 --model_name=winetest \
 --model_base_path=/eltesto)>server.log 2>&1

#CMD ["/usr/sbin/nginx"]
#EXPOSE 8070
