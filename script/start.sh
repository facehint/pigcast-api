source /home/ecs-user/facehint-api/bin/activate
cd /home/ecs-user/app/app
# python -m uvicorn --app-dir /home/ecs-user/app  main:app --reload --host 0.0.0.0 --port 80
python -m uvicorn --app-dir /home/ecs-user/app  main:app --reload --host 0.0.0.0 --port 443 --ssl-keyfile /etc/letsencrypt/live/api.facehint.xyz/privkey.pem --ssl-certfile /etc/letsencrypt/live/api.facehint.xyz/fullchain.pem
