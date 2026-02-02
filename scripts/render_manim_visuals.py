
from manim import *
import pandas as pd
import os

# Settings to make it look like 3b1b
config.background_color = "#1e1e1e" # Dark grey/black
config.pixel_height = 1080
config.pixel_width = 1920

DATA_DIR = os.path.join("data", "results")

class PerformanceOverview(Scene):
    def construct(self):
        # 1. Title
        title = Text("PulmoTrace: Validated Performance", font_size=48, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        
        # 2. Load Data
        df = pd.read_csv(os.path.join(DATA_DIR, "table1_performance_metrics.csv"))
        # Columns: Variable, Unit, R2_Score, RMSE...
        
        # Select key metrics to display
        metrics = [
            ("MMAD", df.loc[df['Variable'] == 'MMAD (Particle Size)', 'R2_Score'].values[0]),
            ("Concentration", df.loc[df['Variable'] == 'Concentration', 'R2_Score'].values[0]),
            ("TB Deposition", df.loc[df['Variable'] == 'Regional Deposition (TB)', 'R2_Score'].values[0]),
            ("ALV Deposition", df.loc[df['Variable'] == 'Regional Deposition (ALV)', 'R2_Score'].values[0]),
        ]
        
        # 3. Create Visuals
        bars = VGroup()
        labels = VGroup()
        scores = VGroup()
        
        max_width = 6.0
        start_y = 2.0
        gap = 1.2
        
        for i, (name, score) in enumerate(metrics):
            # Label
            label = Text(name, font_size=36).to_edge(LEFT, buff=2.0).shift(DOWN * (i * gap) + UP * start_y)
            labels.add(label)
            
            # Bar background
            bg_bar = line = Line(start=label.get_right() + RIGHT*0.5, end=label.get_right() + RIGHT*(0.5+max_width), color=GREY_E, stroke_width=20)
            
            # Active Bar (Animated)
            bar_len = max_width * score
            bar = Line(start=label.get_right() + RIGHT*0.5, end=label.get_right() + RIGHT*(0.5+bar_len), color=TEAL, stroke_width=20)
            bars.add(bar)
            
            # Score Text (Decimal)
            score_text = DecimalNumber(0, num_decimal_places=3, font_size=36)
            score_text.next_to(bg_bar, RIGHT)
            # We want it to count up to 'score'
            scores.add(score_text)
            
            # Add to scene (Backgrounds first instantly)
            self.add(bg_bar)

        self.play(FadeIn(labels, shift=RIGHT))
        
        # Animate Bars and Numbers
        anims = []
        for i, (bar, score_obj, (_, target_score)) in enumerate(zip(bars, scores, metrics)):
            anims.append(Create(bar))
            anims.append(ChangeDecimalToValue(score_obj, target_score))
            
        self.play(*anims, run_time=2.0)
        self.wait(1)
        
        # Add R^2 Label
        r2_label = Text("R² Score (Accuracy)", font_size=24, color=TEAL).next_to(bars[0], UP)
        self.play(FadeIn(r2_label))
        
        self.wait(2)


class AlignmentShowcase(Scene):
    def construct(self):
        # 1. Title
        title = Text("Deconvolution vs Physics: Side-by-Side", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        
        # 2. Load Data
        df = pd.read_csv(os.path.join(DATA_DIR, "table8_side_by_side_alignment.csv"))
        top_samples = df.head(3) # Just show 3 samples nicely
        
        # 3. Setup Layout
        # Creating a "Manim Table" manually with VGroups for animation control
        
        # Headers
        h_sample = Text("Sample", font_size=28, color=BLUE).shift(LEFT*5 + UP*2)
        h_bio = Text("Biological Fraction", font_size=28, color=GREEN).shift(LEFT*1 + UP*2)
        h_phys = Text("Physical Fraction", font_size=28, color=YELLOW).shift(RIGHT*3 + UP*2)
        h_status = Text("Status", font_size=28).shift(RIGHT*6 + UP*2)
        
        headers = VGroup(h_sample, h_bio, h_phys, h_status)
        self.play(FadeIn(headers, shift=DOWN))
        self.play(Create(Line(LEFT*7, RIGHT*7).next_to(headers, DOWN)))
        
        y_pos = 1.0
        for idx, row in top_samples.iterrows():
            # Row Elements
            t_id = Text(row['Sample'], font_size=24).move_to([h_sample.get_x(), y_pos, 0])
            
            # Bio Bar (Bronchial %)
            bio_val = row['Bio_Bronchial']
            bio_bar = Rectangle(width=2.0, height=0.3, color=GREEN, fill_opacity=0.5)
            bio_fill = Rectangle(width=2.0*bio_val, height=0.3, color=GREEN, fill_opacity=1.0).align_to(bio_bar, LEFT)
            bio_group = VGroup(bio_bar, bio_fill).move_to([h_bio.get_x(), y_pos, 0])
            bio_text = Text(f"{bio_val:.2f}", font_size=20).next_to(bio_group, DOWN, buff=0.1)
            
            # Phys Bar (Bronchial %)
            phys_val = row['Phys_Bronchial']
            phys_bar = Rectangle(width=2.0, height=0.3, color=YELLOW, fill_opacity=0.5)
            phys_fill = Rectangle(width=2.0*phys_val, height=0.3, color=YELLOW, fill_opacity=1.0).align_to(phys_bar, LEFT)
            phys_group = VGroup(phys_bar, phys_fill).move_to([h_phys.get_x(), y_pos, 0])
            phys_text = Text(f"{phys_val:.2f}", font_size=20).next_to(phys_group, DOWN, buff=0.1)
            
            # Match Status
            if row['Status'] == 'Match':
                status = Text("✅", font_size=24, color=GREEN)
            else:
                status = Text("❌", font_size=24, color=RED)
            status.move_to([h_status.get_x(), y_pos, 0])
            
            # Animate Row Entrance
            self.play(FadeIn(t_id, shift=RIGHT), run_time=0.5)
            
            # Grow Bars
            self.play(
                GrowFromEdge(bio_group, LEFT),
                GrowFromEdge(phys_group, LEFT),
                run_time=0.8
            )
            self.add(bio_text, phys_text)
            
            # Pop Checkmark
            self.play(SpinInFromNothing(status), run_time=0.4)
            
            y_pos -= 1.5
            
        self.wait(2)
