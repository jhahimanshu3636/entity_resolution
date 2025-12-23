import pandas as pd
from entity_resolution_script import IntegratedEntityResolver
import time
from datetime import datetime
import sys


def main(): 
    print("\n" + "=" * 80)
    print("CORRECTED ENTITY RESOLUTION v4.0")
    print("Single-pass processing with context validation")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    
    INPUT_FILE = '/Users/himanshujha/Desktop/VS_Code/entity_resolution/input_data/export_v2.csv'
    OUTPUT_FILE = 'entity_resolution_CORRECTED_output.xlsx'
    
    SIMILARITY_THRESHOLD = 0.85
    CONTEXT_WEIGHT = 0.25
    
    print(f"⚙️  Configuration:")
    print(f"   Input file: {INPUT_FILE}")
    print(f"   Output file: {OUTPUT_FILE}")
    print(f"   Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"   Context weight: {CONTEXT_WEIGHT} ({CONTEXT_WEIGHT*100:.0f}% context, {(1-CONTEXT_WEIGHT)*100:.0f}% name)")
    print()
    
    # ==========================================================================
    # STEP 1: LOAD DATA
    # ==========================================================================
    
    print("STEP 1/3: Loading data...")
    print("-" * 80)
    
    try:
        encodings = ['ISO-8859-1', 'utf-8', 'latin1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                print(f"   Trying encoding: {encoding}...", end=" ")
                df = pd.read_csv(
                    INPUT_FILE,
                    encoding=encoding,
                    on_bad_lines='skip',
                    low_memory=False
                )
                print("✓")
                break
            except Exception:
                print("✗")
                continue
        
        if df is None:
            raise Exception("Could not read file with any encoding")
        
        print(f"\n✓ Loaded successfully!")
        print(f"   Total rows: {len(df):,}")
        print(f"   Unique companies: {df['buyer-supplier'].nunique():,}")
        print()
        
    except Exception as e:
        print(f"\n✗ Error loading file: {e}")
        return
    
    # ==========================================================================
    # STEP 2: PROCESS ALL DATA IN SINGLE PASS
    # ==========================================================================
    
    print("STEP 2/3: Processing entity resolution...")
    print("-" * 80)
    print(f"   Processing all data together (no partitioning)")
    print()
    
    try:
        resolver = IntegratedEntityResolver(
            similarity_threshold=SIMILARITY_THRESHOLD,
            context_weight=CONTEXT_WEIGHT
        )
        
        result_df, clusters = resolver.resolve_entities(
            df,
            name_column='buyer-supplier',
            transaction_count_column='count'
        )
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user (Ctrl+C)")
        return
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==========================================================================
    # STEP 3: SAVE RESULTS
    # ==========================================================================
    
    print("\nSTEP 3/3: Saving results...")
    print("-" * 80)
    
    # Generate statistics
    total_companies = df['buyer-supplier'].nunique()
    total_clusters = result_df['cluster_id'].nunique()
    multi_member = len(result_df[result_df['cluster_size'] > 1])
    dedup_rate = ((total_companies - total_clusters) / total_companies * 100) if total_companies > 0 else 0
    avg_confidence = result_df['match_confidence'].mean()
    low_conf_count = len(result_df[result_df['match_confidence'] < 0.85])
    
    print(f"\n   📊 Results Summary:")
    print(f"      Input companies: {total_companies:,}")
    print(f"      Unique entities (clusters): {total_clusters:,}")
    print(f"      Companies in multi-member clusters: {multi_member:,}")
    print(f"      Deduplication rate: {dedup_rate:.1f}%")
    print(f"      Average match confidence: {avg_confidence:.3f}")
    print(f"      Low confidence matches (<0.85): {low_conf_count:,}")
    
    # Save to Excel
    print(f"\n   Saving to Excel: {OUTPUT_FILE}...")
    
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            # Sheet 1: Detailed results
            result_df.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # Sheet 2: Cluster summary
            cluster_summary = result_df.groupby('cluster_id').agg({
                'parent_name': 'first',
                'canonical_name': 'first',
                'cluster_size': 'first',
                'match_confidence': 'mean',
                'buyer-supplier': lambda x: ' | '.join(sorted(set(x))[:10])
            }).reset_index()
            
            cluster_summary.columns = ['cluster_id', 'parent_name', 'canonical_name',
                                      'cluster_size', 'avg_confidence', 'sample_variants']
            cluster_summary = cluster_summary.sort_values('cluster_size', ascending=False)
            
            cluster_summary.to_excel(writer, sheet_name='Cluster Summary', index=False)
            
            # Sheet 3: Statistics
            stats_data = {
                'Metric': [
                    'Total Input Rows',
                    'Total Input Companies',
                    'Total Clusters Created',
                    'Multi-Member Clusters',
                    'Deduplication Rate (%)',
                    'Average Match Confidence',
                    'Low Confidence Matches (<0.85)',
                    'Processing Time (minutes)',
                    'Similarity Threshold',
                    'Context Weight'
                ],
                'Value': [
                    len(df),
                    total_companies,
                    total_clusters,
                    len(cluster_summary[cluster_summary['cluster_size'] > 1]),
                    f"{dedup_rate:.1f}",
                    f"{avg_confidence:.3f}",
                    low_conf_count,
                    f"{(time.time() - start_time)/60:.1f}",
                    SIMILARITY_THRESHOLD,
                    CONTEXT_WEIGHT
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Sheet 4: Review queue
            review_queue = cluster_summary[
                (cluster_summary['cluster_size'] > 5) |
                (cluster_summary['avg_confidence'] < 0.90)
            ].head(100)
            
            if len(review_queue) > 0:
                review_queue.to_excel(writer, sheet_name='Review Queue', index=False)
        
        print(f"   ✓ Saved successfully!")
        
    except Exception as e:
        print(f"   ✗ Error saving Excel: {e}")
        print(f"   Saving as CSV instead...")
        csv_file = OUTPUT_FILE.replace('.xlsx', '.csv')
        result_df.to_csv(csv_file, index=False)
        print(f"   ✓ Saved to: {csv_file}")
    
    # ==========================================================================
    # COMPLETION SUMMARY
    # ==========================================================================
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ ENTITY RESOLUTION COMPLETE!")
    print("=" * 80)
    
    print(f"\n⏱️  Processing Statistics:")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    print(f"\n📁 Output Files:")
    print(f"   • {OUTPUT_FILE}")
    print(f"   • {OUTPUT_FILE.replace('.xlsx', '.csv')}")
    
    print(f"\n📊 Quality Metrics:")
    print(f"   • Deduplication rate: {dedup_rate:.1f}%")
    print(f"   • Average confidence: {avg_confidence:.3f}")
    print(f"   • Clusters requiring review: {len(review_queue) if len(review_queue) > 0 else 0}")
    
    if dedup_rate < 5:
        print(f"\n⚠️  Low deduplication rate ({dedup_rate:.1f}%)")
        print(f"   Consider lowering SIMILARITY_THRESHOLD to 0.80")
    elif dedup_rate > 20:
        print(f"\n⚠️  High deduplication rate ({dedup_rate:.1f}%)")
        print(f"   Review results carefully for false positives")
        print(f"   Consider increasing SIMILARITY_THRESHOLD to 0.87")
    else:
        print(f"\n✓ Deduplication rate looks reasonable")
    
    print(f"\n{' ' * 80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)