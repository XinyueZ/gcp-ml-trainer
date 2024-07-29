#run 
python pipeline.py --key_dir "OAuth2" --project_id "isochrone-isodistance" --predefined_acl "projectPrivate" --location "europe-west1" --model_name "gemini-chat"
python pipeline.py --key_dir "OAuth2" --project_id "isochrone-isodistance" --predefined_acl "projectPrivate" --location "europe-west1" --model_name "gemma-instruct"
python pipeline.py --key_dir "OAuth2" --project_id "isochrone-isodistance" --predefined_acl "projectPrivate" --location "europe-west1" --model_name "chat-bison"  
python pipeline.py --key_dir "OAuth2" --project_id "isochrone-isodistance" --predefined_acl "projectPrivate" --location "europe-west1" --model_name "text-bison"  
