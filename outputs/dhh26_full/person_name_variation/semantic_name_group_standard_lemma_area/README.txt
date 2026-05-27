Semantic name group -> standard lemma by area

Grouping rule:
1. semantic_name_group is assigned from cleaned word_in_english only, using the same strict classifier as word_in_english_only_name_variant_area_counts.csv.
2. Within each semantic_name_group, regional variants are standard_lemma values from word_analysis.parquet.
3. *_raw.csv keeps every standard lemma; *_for_map.csv collapses each feature to its top 8 standard lemmas plus Other so maps have manageable legends.
4. feature_rank_by_size orders semantic groups by mapped mention count.

Main map-ready file:
/Users/xiaoye/Documents/New project/outputs/dhh26_full/person_name_variation/semantic_name_group_standard_lemma_area/semantic_name_group_standard_lemma_area_counts_for_map.csv

Dominance file:
/Users/xiaoye/Documents/New project/outputs/dhh26_full/person_name_variation/semantic_name_group_standard_lemma_area/semantic_name_group_standard_lemma_area_dominance.csv