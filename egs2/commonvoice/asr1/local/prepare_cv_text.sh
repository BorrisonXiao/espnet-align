dir="data"

mv "$dir/train_zh_HK/text" "$dir/train_zh_HK/text.raw"
mv "$dir/test_zh_HK/text" "$dir/test_zh_HK/text.raw"
mv "$dir/dev_zh_HK/text" "$dir/dev_zh_HK/text.raw"
mv "$dir/validated_zh_HK/text" "$dir/validated_zh_HK/text.raw"

# Perform word segmentation
python local/cantonese_text_process.py --input "$dir/train_zh_HK/text.raw" --output "$dir/train_zh_HK/text" --word_seg --commonvoice
python local/cantonese_text_process.py --input "$dir/test_zh_HK/text.raw" --output "$dir/test_zh_HK/text" --word_seg --commonvoice
python local/cantonese_text_process.py --input "$dir/dev_zh_HK/text.raw" --output "$dir/dev_zh_HK/text" --word_seg --commonvoice
python local/cantonese_text_process.py --input "$dir/validated_zh_HK/text.raw" --output "$dir/validated_zh_HK/text" --word_seg --commonvoice