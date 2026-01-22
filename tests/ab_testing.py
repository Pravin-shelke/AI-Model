"""
A/B TESTING FRAMEWORK
Compare AI predictions vs manual entry to measure:
- Time savings
- Accuracy
- User acceptance rates
"""

import json
import time
from datetime import datetime
import random


class ABTestFramework:
    """Framework for A/B testing AI predictions in production"""
    
    def __init__(self, test_ratio=0.1):
        """
        Initialize A/B testing framework
        
        Args:
            test_ratio: Percentage of users to receive AI predictions (0.0-1.0)
        """
        self.test_ratio = test_ratio
        self.test_results = []
    
    def assign_user_to_group(self, user_id):
        """
        Randomly assign user to control or test group
        
        Returns:
            'test': User gets AI predictions
            'control': User fills manually (traditional flow)
        """
        # Use hash of user_id for consistent assignment
        random.seed(hash(user_id))
        return 'test' if random.random() < self.test_ratio else 'control'
    
    def start_session(self, user_id, group):
        """Start tracking a user session"""
        return {
            'session_id': f"{user_id}_{datetime.now().timestamp()}",
            'user_id': user_id,
            'group': group,
            'start_time': time.time(),
            'predictions_shown': 0,
            'predictions_accepted': 0,
            'predictions_edited': 0,
            'predictions_rejected': 0,
            'questions_answered': 0,
            'total_questions': 0
        }
    
    def track_prediction(self, session, indicator, predicted_value, user_action, final_value):
        """
        Track user interaction with AI prediction
        
        Args:
            session: Current session dict
            indicator: Question indicator
            predicted_value: AI predicted value
            user_action: 'accepted', 'edited', 'rejected'
            final_value: Final value user submitted
        """
        session['predictions_shown'] += 1
        
        if user_action == 'accepted':
            session['predictions_accepted'] += 1
        elif user_action == 'edited':
            session['predictions_edited'] += 1
        elif user_action == 'rejected':
            session['predictions_rejected'] += 1
        
        return {
            'indicator': indicator,
            'predicted_value': predicted_value,
            'user_action': user_action,
            'final_value': final_value,
            'was_correct': predicted_value == final_value,
            'timestamp': datetime.now().isoformat()
        }
    
    def end_session(self, session):
        """Complete session and calculate metrics"""
        session['end_time'] = time.time()
        session['duration_seconds'] = session['end_time'] - session['start_time']
        session['duration_minutes'] = session['duration_seconds'] / 60
        
        # Calculate acceptance rate
        if session['predictions_shown'] > 0:
            session['acceptance_rate'] = session['predictions_accepted'] / session['predictions_shown']
            session['edit_rate'] = session['predictions_edited'] / session['predictions_shown']
            session['rejection_rate'] = session['predictions_rejected'] / session['predictions_shown']
        else:
            session['acceptance_rate'] = 0
            session['edit_rate'] = 0
            session['rejection_rate'] = 0
        
        # Save result
        self.test_results.append(session)
        
        # Log to file
        with open('ab_test_results.jsonl', 'a') as f:
            f.write(json.dumps(session) + '\n')
        
        return session
    
    def analyze_results(self):
        """Analyze A/B test results"""
        if not self.test_results:
            return {'message': 'No test results available yet'}
        
        import pandas as pd
        df = pd.DataFrame(self.test_results)
        
        # Split by group
        control = df[df['group'] == 'control']
        test = df[df['group'] == 'test']
        
        results = {
            'generated_at': datetime.now().isoformat(),
            'total_sessions': len(df),
            'control_group': {
                'sessions': len(control),
                'avg_duration_minutes': control['duration_minutes'].mean() if len(control) > 0 else 0,
                'median_duration_minutes': control['duration_minutes'].median() if len(control) > 0 else 0
            },
            'test_group': {
                'sessions': len(test),
                'avg_duration_minutes': test['duration_minutes'].mean() if len(test) > 0 else 0,
                'median_duration_minutes': test['duration_minutes'].median() if len(test) > 0 else 0,
                'avg_acceptance_rate': test['acceptance_rate'].mean() if len(test) > 0 else 0,
                'avg_predictions_shown': test['predictions_shown'].mean() if len(test) > 0 else 0
            }
        }
        
        # Calculate impact
        if len(control) > 0 and len(test) > 0:
            time_saved_minutes = control['duration_minutes'].mean() - test['duration_minutes'].mean()
            time_saved_percent = (time_saved_minutes / control['duration_minutes'].mean()) * 100
            
            results['impact'] = {
                'time_saved_minutes': time_saved_minutes,
                'time_saved_percent': time_saved_percent,
                'statistical_significance': self._check_significance(control, test)
            }
        
        return results
    
    def _check_significance(self, control, test):
        """
        Simple statistical significance check
        (In production, use proper statistical tests like t-test)
        """
        try:
            from scipy import stats
            
            t_stat, p_value = stats.ttest_ind(
                control['duration_minutes'].dropna(),
                test['duration_minutes'].dropna()
            )
            
            is_significant = p_value < 0.05
            
            return {
                'p_value': p_value,
                'is_significant': is_significant,
                'interpretation': 'Statistically significant' if is_significant else 'Not significant'
            }
        except:
            return {'message': 'Statistical test not available'}
    
    def generate_report(self, output_file='ab_test_report.json'):
        """Generate comprehensive A/B test report"""
        results = self.analyze_results()
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 A/B TEST RESULTS")
        print("=" * 70)
        
        if 'control_group' in results:
            print(f"\n🔵 Control Group (Manual Entry):")
            print(f"  Sessions: {results['control_group']['sessions']}")
            print(f"  Avg Duration: {results['control_group']['avg_duration_minutes']:.1f} minutes")
            
            print(f"\n🟢 Test Group (AI Predictions):")
            print(f"  Sessions: {results['test_group']['sessions']}")
            print(f"  Avg Duration: {results['test_group']['avg_duration_minutes']:.1f} minutes")
            print(f"  Acceptance Rate: {results['test_group']['avg_acceptance_rate']:.1%}")
            
            if 'impact' in results:
                print(f"\n⚡ Impact:")
                print(f"  Time Saved: {results['impact']['time_saved_minutes']:.1f} minutes")
                print(f"  Time Saved %: {results['impact']['time_saved_percent']:.1f}%")
                
                if results['impact']['statistical_significance'].get('is_significant'):
                    print(f"  ✅ Statistically significant (p < 0.05)")
                else:
                    print(f"  ⚠️  Not statistically significant yet - need more data")
        
        print("\n" + "=" * 70)
        
        return results


def example_usage():
    """Example of how to use A/B testing framework"""
    
    # Initialize framework (10% of users get AI)
    ab_test = ABTestFramework(test_ratio=0.1)
    
    # Simulate user session
    user_id = "user123"
    group = ab_test.assign_user_to_group(user_id)
    
    print(f"User {user_id} assigned to: {group}")
    
    # Start session
    session = ab_test.start_session(user_id, group)
    
    if group == 'test':
        # User gets AI predictions
        print("\nShowing AI predictions...")
        
        # Track interactions
        ab_test.track_prediction(
            session,
            indicator='BH-1',
            predicted_value='Yes',
            user_action='accepted',
            final_value='Yes'
        )
        
        ab_test.track_prediction(
            session,
            indicator='BH-2',
            predicted_value='No',
            user_action='edited',
            final_value='Yes'
        )
    
    # End session
    session['questions_answered'] = 150
    session['total_questions'] = 266
    
    result = ab_test.end_session(session)
    
    print(f"\nSession completed in {result['duration_minutes']:.2f} minutes")
    if group == 'test':
        print(f"Acceptance rate: {result['acceptance_rate']:.1%}")
    
    # Generate report
    ab_test.generate_report()


if __name__ == "__main__":
    example_usage()
