# Using the official tensorflow serving image from docker hub as base image
FROM ubuntu:18.04
RUN apt-get install tensorflow-model-server
# Installing NGINX
RUN apt-get update && apt-get install -y --no-install-recommends nginx

# Copy our model folder to the container
COPY eltesto /eltesto

# Copy NGINX configuration to the container
COPY nginx.conf /etc/nginx/nginx.conf



# starts NGINX and TF serving pointing to our model
ENTRYPOINT (service nginx start) | (tensorflow_model_server --rest_api_port=5000 \
 --model_name=winetest \
 --model_base_path=/eltesto)>server.log 2>&1
#EXPOSE 8070
