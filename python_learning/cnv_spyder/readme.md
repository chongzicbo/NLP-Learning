# pmcid全文文献爬取
nohup python file_spyder.py --save_dir /mnt/B2C_USER/chengbo1/data/cnv --input_file_list /mnt/B2C_USER/chengbo1/data/cnv_pmcid_30000.csv --log_file cnv_pmcid_30000.log > /mnt/B2C_USER/chengbo1/data/logs/cnv_pmcid_30000.log 2>&1 &

# pmid摘要文献爬取
nohup python pmid_xml.py --save_dir /mnt/B2C_USER/chengbo1/data/cnv/pmid --input_file_list /mnt/B2C_USER/chengbo1/data/all_cnv_pmid_nopmcid_list-20220704.csv --log_file all_cnv_pmid_nopmcid_list.log > /mnt/B2C_USER/chengbo1/data/logs/all_cnv_pmid_nopmcid_list.log 2>&1 &