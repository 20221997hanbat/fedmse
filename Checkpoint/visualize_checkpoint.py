import os
import glob
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_latent(base_dir, output_base_dir):
    print("\n--- [1] LatentData 시각화 (PCA) 시작 ---")
    latent_search_path = os.path.join(base_dir, "LatentData", "**", "*.pkl")
    pkl_files = glob.glob(latent_search_path, recursive=True)

    if not pkl_files:
        print("[!] LatentData 폴더에서 시각화할 .pkl 파일을 찾지 못했습니다.")
        return

    for pkl_path in pkl_files:
        rel_path = os.path.relpath(pkl_path, base_dir)
        rel_dir = os.path.dirname(rel_path)
        file_name = os.path.splitext(os.path.basename(pkl_path))[0]

        target_dir = os.path.join(output_base_dir, rel_dir)
        os.makedirs(target_dir, exist_ok=True)
        save_png_path = os.path.join(target_dir, f"vis_{file_name}.png")

        print(f"[처리 중] {rel_path}")

        try:
            with open(pkl_path, "rb") as f:
                latent_data = pickle.load(f)

            if not isinstance(latent_data, dict) or not latent_data:
                print("  └─ [스킵] 지원하지 않는 데이터 형식입니다.")
                continue

            last_round = max(latent_data.keys())
            client_dict = latent_data[last_round]
            first_client = list(client_dict.keys())[0]
            test_latent, test_label = client_dict[first_client]

            if hasattr(test_latent, "cpu"): test_latent = test_latent.cpu().numpy()
            if hasattr(test_label, "cpu"): test_label = test_label.cpu().numpy()

            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(test_latent)

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=test_label, cmap='coolwarm', alpha=0.6, s=15)
            plt.colorbar(scatter, label='Class (0: Normal, 1: Attack)')
            plt.title(f"Latent Space Visualization\nClient: {first_client} (Round {last_round+1})")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.grid(True)

            plt.savefig(save_png_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  └─ 생성 완료: {save_png_path}")

        except Exception as e:
            print(f"  └─ [오류 발생] {file_name} 시각화 실패: {e}")

def visualize_json_results(base_dir, output_base_dir):
    print("\n--- [2] Results 시각화 (AUC Trend & Bar) 시작 ---")
    results_dir = os.path.join(base_dir, "Results")
    json_pattern = os.path.join(results_dir, "**", "*.json")
    json_files = glob.glob(json_pattern, recursive=True)

    if not json_files:
        print("[!] Results 폴더에서 .json 파일을 찾지 못했습니다.")
        return

    for json_path in json_files:
        rel_path = os.path.relpath(json_path, base_dir)
        rel_dir = os.path.dirname(rel_path)
        file_name = os.path.splitext(os.path.basename(json_path))[0]

        target_dir = os.path.join(output_base_dir, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

        print(f"[처리 중] {rel_path}")

        try:
            aggregated = {}
            
            # JSON Lines 형태의 파일을 한 줄씩 읽어서 파싱
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    round_data = json.loads(line)
                    # {"round_1": {"client-1": 0.99, "global_loss": 0.44, "join_clients": [...]}} 형태 처리
                    for round_key, metrics in round_data.items():
                        if isinstance(metrics, dict):
                            for metric_name, metric_val in metrics.items():
                                # 수치형 데이터(AUC, loss 등)만 그래프 용도로 수집 (join_clients 같은 리스트는 제외)
                                if isinstance(metric_val, (int, float)):
                                    if metric_name not in aggregated:
                                        aggregated[metric_name] = []
                                    aggregated[metric_name].append(metric_val)

            line_data = {}
            bar_data = {}

            # 그래프 그리기용 데이터 분류
            for key, value in aggregated.items():
                if len(value) > 1:
                    line_data[key] = value
                    bar_data[key] = value[-1]  # 막대 그래프용으로는 마지막 라운드(최종) 값 사용
                elif len(value) == 1:
                    bar_data[key] = value[0]

            # 1. 선 그래프 (Trend)
            if line_data:
                plt.figure(figsize=(12, 7))
                for k, v in line_data.items():
                    # global_loss는 AUC와 스케일이 다르므로 선 그래프에서 제외하거나 따로 그릴 수 있지만, 
                    # 일단 모두 표출하되 점선 등 스타일을 다르게 할 수도 있습니다. 여기서는 기본 형태로 그립니다.
                    if 'loss' in k.lower():
                        plt.plot(range(1, len(v) + 1), v, marker='x', linestyle='--', label=k)
                    else:
                        plt.plot(range(1, len(v) + 1), v, marker='o', label=k)
                        
                plt.title(f"Performance Trend per Round\n({file_name})")
                plt.xlabel("Round")
                plt.ylabel("Score / Loss")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 범례가 겹치지 않게 바깥으로 이동
                plt.grid(True)
                plt.tight_layout()
                
                save_path = os.path.join(target_dir, f"trend_{file_name}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"  └─ 선 그래프 생성 완료")

            # 2. 막대 그래프 (Bar)
            # bar 그래프에는 loss를 제외하고 AUC(정확도) 계열만 표시하는 것이 깔끔합니다.
            auc_bar_data = {k: v for k, v in bar_data.items() if 'loss' not in k.lower()}
            
            if auc_bar_data:
                plt.figure(figsize=(10, 6))
                names = list(auc_bar_data.keys())
                values = list(auc_bar_data.values())
                
                bars = plt.bar(names, values, color='#4c72b0')
                plt.ylim(0, min(max(values) * 1.1, 1.05) if values else 1.0)
                
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=8)
                
                plt.title(f"Final AUC Performance\n({file_name})")
                plt.ylabel("AUC Score")
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                save_path = os.path.join(target_dir, f"bar_{file_name}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"  └─ 막대 그래프 생성 완료")

            if not line_data and not bar_data:
                print("  └─ [스킵] 시각화할 수치 데이터가 없습니다.")

        except Exception as e:
            print(f"  └─ [오류 발생] 시각화 실패: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(base_dir, "Visualized")
    
    print("=== 통합 시각화 스크립트 실행 ===")
    visualize_latent(base_dir, output_base_dir)
    visualize_json_results(base_dir, output_base_dir)
    print("\n=== 모든 이미지 생성이 완료되었습니다! ===")
    print(f"이미지 저장 경로: {output_base_dir}")

if __name__ == "__main__":
    main()