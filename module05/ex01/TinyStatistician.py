import math

class TinyStatistician(object):
    
    @staticmethod
    def mean(x):
        """
        Returns the mean of a list or array
        """
        if type(x) == list and len(x) > 0:
            result = 0
            for num in x:
                result += num
            return result / len(x)
        return None
    
    @staticmethod
    def median(x):
        """
        Returns the median of a list or array
        """
        if type(x) == list and len(x) > 0:
            return TinyStatistician().percentile(x, 50)
        return None

    @staticmethod
    def quartile(x):
        """
        Returns the quartile of a list or array
        """
        if type(x) == list and len(x) > 0:
            return [TinyStatistician().percentile(x, 25), TinyStatistician().percentile(x, 75)]
        return None

    @staticmethod
    def percentile(x, p):
        """
        Returns the given percentile of a list or array
        """
        if type(x) == list and len(x) > 0:
            i = (p / 100) * len(x)
            sorted_x = x.copy()
            sorted_x.sort()
            if i - int(i) == 0:
                return (sorted_x[int(i) - 1] + sorted_x[int(i)]) / 2
            return float(sorted_x[int(i)])
        return None

    @staticmethod
    def var(x):
        """
        Returns the variance of a list or array
        """
        if type(x) == list and len(x) > 0:
            variance = 0.0
            mean = TinyStatistician().mean(x)
            for num in x:
                variance += (num - mean)**2
            return variance / len(x)
        return None

    @staticmethod
    def std(x):
        """
        Returns the standard variation of a list or array
        """
        if type(x) == list and len(x) > 0:
            return math.sqrt(TinyStatistician().var(x))
        return None
