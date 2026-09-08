llm_name=$1
test_ratio=${2:-0.05}

python inference.py --llm_name "$llm_name" --prop 'LogP'      --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'TPSA'      --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'QED'       --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&TPSA' --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&QED'  --test_ratio "$test_ratio" --accuracy_only
