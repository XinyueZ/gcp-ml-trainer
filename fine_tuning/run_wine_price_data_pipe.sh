#run 
python wine_price_data_pipe.py --model_name "gemini-chat" --project_id "isochrone-isodistance" --predefined_acl "projectPrivate"
python wine_price_data_pipe.py --model_name "gemma"       --project_id "isochrone-isodistance" --predefined_acl "projectPrivate"
python wine_price_data_pipe.py --model_name "chat-bison"  --project_id "isochrone-isodistance" --predefined_acl "projectPrivate"
python wine_price_data_pipe.py --model_name "text-bison"  --project_id "isochrone-isodistance" --predefined_acl "projectPrivate"
 
