# Igor Golubenov
## Junior Frontend Developer

# How to contact me?

* Phone/Telegram: +375293803777
* Email: igorgworking@gmail.com

# My Goals
My professional goal is to create excellent products that make people happier, more productive, creative and are available to everyone without exceptions. I have a very diverse experience in IT-industry, spanning from Data Science to UX/UI design. I'm fascinated by exploring new areas, acquiring skills and finding my perfect job! After a long run of backend development I've decided to learn Frontend to work closer to human-visible side of the applications.

# My experience
My previous position was a **data analyst** at a retail company. I was attached to different projects within the company, creating reports for different departments. I have implemented non-liquid items detection system, which automatically downloads data from internal ERP-system server via SMB-protocol and performs analysis using Python libraries (pandas and numpy). This solution is integrated in company’s catalogue.
Other project included creating marketing reports, using data from very different sources: **Google Analytics, Google AdWords, Yandex.Market, Yandex.Direct, Facebook**, company’s site and internal ERP-system. The date was collected and loaded in **Google BigQuery** tables with the help of another company’s solution. My job was to correctly merge the data and create appropriate reports with visualizations. Merging was performed with the help of SQL right in BigQuery, visualization – using **Google Data Studio**.

My previous reports have been created in **Power BI((. Among different retrospective charts, I tried to plot predictions for company’ s sales time series for few next weeks, though the solution was not widely used due to difficulties with cleaning data and having different sales channels.
Back in university I was studying **Multivariate Statistical Analysis, Econometrics** and have some experience in time series analysis. We studied all classic models **(AR, MA, ARMA, ARIMA)**, ways to decompose time series in trend, seasonal components, model accuracy metrics.

Also I have pretty wide experience in **Machine Learning**. My latter interest is Computer Vision. I completed an online course focused on Deep Learning, it’s mathematical back-end, implementations and tricks for applied problems. Right now I participate in Kaggle Competition, which goal is Bengali handwritten graphemes recognition. I have already tried an implementation of **ResNet** on the dataset, which showed pretty good accuracy (around 0.96). 
During my CV course I have implemented **Multilayer Perceptron** and **Convolutional** layer myself with analytical and computed gradient comparison as a sanity check (they were equal). For further tasks I switched to **PyTorch** and **Tensorflow**, so that I could focus on model’s architecture and hyperparameter tuning. I successfully used **transfer learning** technique to change output layer and switch from ImageNet classification problem to binary classification. I also became familiar with Apache Spark interface during another course focused on parallel computing.

My experience in Python includes using **Django** for personal site with a database **(sqlite)**. The other site I created was used for deploying trained ML model (**Random Forest** for a regression problem). The model implementation was taken from **scikit-learn** library and serialized using **pickle**.
Other Computer Science areas and programming languages I’ve worked with: **Java, JavaScript, C++, Linux, bash scripts, Docker, Selenium, R, Git, SQL, SAP FIORI Framework, HTML/CSS, REST architecture, Database normalization, Data Structures**.

# Code Examples

## Knapsack problem solving in C++
```C++
#include <iostream> 
#include <list>
#include <vector>
#include <iterator> 
#include <fstream>
#include <string>
#include <algorithm> 
#include <cmath>
#include <bitset>
#include <map>
using namespace std;

struct Combo {
	long long num;
	long long w;
	long long c;
	long long best;
};

vector<Combo>::const_iterator binarySearch(const vector<Combo>& container, long long element)
{
	const vector<Combo>::const_iterator endIt = end(container) - 1;

	vector<Combo>::const_iterator left = begin(container);
	vector<Combo>::const_iterator right = endIt;

	if (container.size() == 0
		|| container.front().w < element)
	{
		return left;
	}

	if (container.back().w > element)
	{
		return right;
	}

	while (distance(left, right) > 0) {
		const vector<Combo>::const_iterator mid = left + distance(left, right) / 2;

		if (element >= (*mid).w) {
			right = mid;
		}
		else {
			left = mid + 1;
		}
	}

	if ((*right).w <= element) {
		return left;
	}

	return endIt;
}

long long const MAXN = 40;
long long n, W;
long long p[MAXN], w[MAXN];
vector<long long> out;

bool knapsack_sorter(const Combo& lhs, const Combo& rhs)
{
	return lhs.w < rhs.w;
}

bool knapsack_sorter_value(long long rhs, const Combo& lhs)
{
	return lhs.w > rhs;
}

int main()
{
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	cin >> n >> W;
	for (long long i = 0; i < n; i++)
	{
		cin >> w[i] >> p[i];
	}

	long long sn = n / 2;
	long long fn = n - sn;
	long long first_size = pow(2, sn) - 1;
	long long second_size = pow(2, fn) - 1;
	vector<Combo> first_vec;
	//Combo *first = new Combo[first_size];
	bitset<MAXN> mask;
	long long i, j;
	for (i = 0; i <= first_size; i++)
	{
		bitset<MAXN> mask(i);
		Combo new_combo;
		new_combo.num = i;
		new_combo.w = 0;
		new_combo.c = 0;
		for (j = 0; j < sn; j++)
		{
			if (mask.test(j))
			{
				new_combo.w += w[j];
				new_combo.c += p[j];
			}
		}
		first_vec.push_back(new_combo);
	}

	sort(first_vec.begin(), first_vec.end(), &knapsack_sorter);
	//vector <Combo>::iterator mit1, mit2;

	long long current_first_part_best_index = 0;
	for (i = 0; i <= first_size; i++) {
		if (first_vec[i].w > W)
			break;

		if (first_vec[i].c > first_vec[current_first_part_best_index].c) {
			current_first_part_best_index = i;
		}

		first_vec[i].best = current_first_part_best_index;
	}
	/*
	for (mit1 = first_vec.begin(); mit1 != first_vec.end(); )
	{
		bool skip = false;
		for (mit2 = mit1 + 1; mit2 != first_vec.end(); mit2++)
		{
			if ((*mit2).w <= (*mit1).w && (*mit2).c >= (*mit1).c)
			{
				mit1 = first_vec.erase(mit1);
				skip = true;
				break;
			}
		}
		if (!skip) mit1++;
	}
	*/
	Combo best_combo;
	//Combo second_best;
	best_combo.num = 0;
	best_combo.w = 0;
	best_combo.c = 0;
	map <long long, long long> second_vec;
	//vector<long long> second_vec;
	//second_best.num = 0;
	//second_best.w = 0;
	//second_best.c = 0;
	for (i = 0; i <= second_size; i++)
	{
		bitset<MAXN> mask(i);
		Combo curr_combo;
		curr_combo.num = i;
		curr_combo.w = 0;
		curr_combo.c = 0;
		for (j = 0; j < fn; j++)
			if (mask.test(j))
			{
				curr_combo.w += w[j + sn];
				curr_combo.c += p[j + sn];
			}
		//vector <Combo>::const_iterator candidate;
		//candidate = binarySearch(first_vec, W - curr_combo.w);

		std::map<long long, long long>::iterator mit = second_vec.find(curr_combo.w);
		if (mit != second_vec.end() && curr_combo.c <= mit->second && curr_combo.w != 0)
			continue;
		auto first_best = upper_bound(first_vec.begin(), first_vec.end(), W - curr_combo.w, knapsack_sorter_value);
		if (first_best != first_vec.begin()) 
			first_best -= 1;
		Combo candidate = first_vec[(*first_best).best];
		if (candidate.w <= W - curr_combo.w && candidate.c + curr_combo.c > best_combo.c)
		{
			best_combo.num = candidate.num + i * pow(2, sn);
			best_combo.w = curr_combo.w + candidate.w;
			best_combo.c = curr_combo.c + candidate.c;
		}
		/*
		second_best.w = curr_combo.w;
		second_best.c = curr_combo.c;
		second_best.num = curr_combo.num;
		*/
		if(mit == second_vec.end())
			second_vec.insert(pair<long long, long long>(curr_combo.w, curr_combo.c));
		else
			mit->second = curr_combo.c;
	}
	bitset<MAXN> answer(best_combo.num);
	for (j = 0; j < MAXN; j++)
		if (answer.test(j))
		{
			out.push_back(j + 1);
		}
	cout << out.size() << endl;
	copy(out.begin(), out.end(), ostream_iterator<long long>(cout, " "));
	return 0;

}
```

## Random Forest Regressor for Django app
```Python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from nltk.corpus import stopwords

stop = stopwords.words('english')

df = pd.read_csv('sample.csv', parse_dates=['время'], engine='python')
df['время'] = df['время'].dt.hour
df.head

if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X, y = df.iloc[:, 0:9].values, df.iloc[:, 11].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
	
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', 
                               random_state=1, 
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print(y_train_pred)
print(y_train)

print(y_test_pred)
print(y_test)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

dest = os.path.join('scooterpredictor', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)   
pickle.dump(forest, open(os.path.join(dest, 'predictor.pkl'), 'wb'), protocol=4)
```

# Foreign language skills:
English(C1), Deutsch(A1)

# Education:
Belarussian State University, Faculty of Applied Math, Speciality: Economical Cybernetics. Year of graduation: 2019
Additional education, certificates:

1. "Introduction to Financial Engineering" (Compatibl);
2. "Design and development of modern analytical systems. DWH. Big Data. Cloud" (IBA Group);
3. "Machine Learning" (Coursera, Andrew Ng);
4. "Modern SAP Development" (EPAM Systems);
5. "Einstein Analytics Data Preparation Specialist" (SalesForce trailhead);
6. "Fundamentals of Scalable Data Science" (IBM, Coursera);
7. "UX/UI Design" (TeachMeSkills)
