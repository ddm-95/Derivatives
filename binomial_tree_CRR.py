import numpy as np


class EuropeanOption_CRR:
    """
    A class to represent an European option priced using the Cox-Ross-Rubinstein (CRR) binomial tree method.

    Attributes
    -----------
    S0 : float
        Initial stock price.
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option in years.
    r : float
        Annual risk-free interest rate.
    sigma : float
        Volatility.
    N : int
        Number of time steps in the binomial tree.
    opt_type : str, optional
        Type of the option, 'Call' or 'Put'. Default is 'Call'.
    vol_shift : float, optional
        Volatility shift used for vega calculation. Default is 0.01.
    r_shift : float, optional
        Interest rate shift used for rho calculation. Default is 0.01.

    Methods
    -----------
    price_and_greeks
        returns the option's price and Greeks and the option's and the underlying's price trees.
    _calculate_option_price
        Calculates the option's price and the option's and the underlying's price trees.
    _calculate_greeks
        Calculates the Greeks of the option.
    summary_metrics
        Prints a summary of option metrics along with the input parameters.
    """

    def __init__(self, S0, K, T, r, sigma, N, opt_type="Call", vol_shift=0.01, r_shift=0.01):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.opt_type = opt_type
        self.vol_shift = vol_shift
        self.r_shift = r_shift

        # Private attributes - can't be changed manually
        self._option_price = None
        self._delta = None
        self._gamma = None
        self._vega = None
        self._theta = None
        self._rho = None
        self._stock_tree = None
        self._option_tree = None

        self._calculate_option_price_and_greeks()

    def _calculate_option_price_and_greeks(self):
        """
        Calculate the option price and Greeks using the CRR binomial tree method.

        Returns:
        --------
        option_price : float
            The price of the option calculated using the CRR binomial tree method.
        delta : float
            The delta of the option.
        gamma : float
            The gamma of the option.
        vega : float
            The vega of the option.
        theta : float
            The theta of the option.
        rho : float
            The rho of the option.
        stock_tree : numpy.ndarray
            A 2D array representing the entire binomial tree of stock prices.
        option_tree : numpy.ndarray
            A 2D array representing the entire binomial tree of option values.
        """
        self.option_price, self.stock_tree, self.option_tree = self._calculate_option_price()
        self.delta, self.gamma, self.vega, self.theta, self.rho = self._calculate_greeks(
            self.stock_tree, self.option_tree)
        return self.option_price, self.delta, self.gamma, self.vega, self.theta, self.rho, self.stock_tree, self.option_tree

    def _calculate_option_price(self):
        """
        Calculate the option price using the CRR binomial tree method.

        Returns:
        --------
        option_price : float
            The price of the option calculated using the CRR binomial tree method.
        stock_tree : numpy.ndarray
            A 2D array representing the entire binomial tree of stock prices.
        option_tree : numpy.ndarray
            A 2D array representing the entire binomial tree of option values.
        """
        # calculated constants
        dt = self.T / self.N  # time intervals
        u = np.exp(self.sigma * np.sqrt(dt))  # up factor
        d = np.exp(-self.sigma * np.sqrt(dt))  # down factor
        # risk neutral probability of an up move
        q = (np.exp(self.r*dt) - d) / (u - d)
        N = self.N + 1

        # initialize arrays to store stock prices and option values
        stock_tree = np.zeros((N, N))
        option_tree = np.zeros((N, N))

        # initialize the first stock price
        stock_tree[0, 0] = self.S0

        # compute stock prices at each node
        for n in range(1, N):
            stock_tree[n, :n] = stock_tree[n-1, :n] * u
            stock_tree[n, n] = stock_tree[n-1, n-1] * d

        # compute option values at maturity
        if self.opt_type == "Call":
            option_tree[-1, :] = np.maximum(stock_tree[-1, :] - self.K, 0)
        else:
            option_tree[-1, :] = np.maximum(self.K - stock_tree[-1, :], 0)

        # compute option values at each previous node using backward induction
        for j in range(N-2, -1, -1):
            for i in range(j+1):
                option_tree[j, i] = np.exp(-self.r * dt) * (
                    q * option_tree[j+1, i] + (1 - q) * option_tree[j+1, i+1])

        option_price = option_tree[0, 0]
        return option_price, stock_tree, option_tree

    def _calculate_greeks(self, stock_tree, option_tree):
        """
        Calculate the Greeks of the option.

        Parameters:
        -----------
        stock_tree : numpy.ndarray
            A 2D array representing the entire binomial tree of stock prices.
        option_tree : numpy.ndarray
            A 2D array representing the entire binomial tree of option values.

        Returns:
        --------
        delta : float
            The delta of the option.
        gamma : float
            The gamma of the option.
        vega : float
            The vega of the option.
        theta : float
            The theta of the option.
        rho : float
            The rho of the option.
        """
        dt = self.T / self.N
        option_price = option_tree[0, 0]

        # Calculate Delta
        delta = (option_tree[1, 0] - option_tree[1, 1]) / \
            (stock_tree[1, 0] - stock_tree[1, 1])

        # Calculate Gamma
        gamma = ((option_tree[2, 0] - option_tree[2, 1]) / (stock_tree[2, 0] - stock_tree[2, 1]) -
                 (option_tree[2, 1] - option_tree[2, 2]) / (stock_tree[2, 1] - stock_tree[2, 2])) / \
                ((stock_tree[2, 0] - stock_tree[2, 1] +
                 stock_tree[2, 1] - stock_tree[2, 2]) / 2)

        # Adjust self.sigma for vega calculation
        orig_sigma = self.sigma
        self.sigma -= self.vol_shift

        # Calculate Vega
        option_price_vshift = self._calculate_option_price()[0]
        vega = (option_price - option_price_vshift) / self.vol_shift

        # Restore original sigma
        self.sigma = orig_sigma

        # Calculate Theta
        theta = (option_tree[2, 1] - option_tree[0, 0]) / (2 * dt)

        # Adjust self.r for rho calculation
        orig_r = self.r
        self.r -= self.r_shift

        # Calculate Rho
        option_price_rshift = self._calculate_option_price()[0]
        rho = (option_price - option_price_rshift) / self.r_shift

        # Restore original r
        self.r = orig_r

        return delta, gamma, vega, theta, rho

    def _print_summary_metrics(self):
        """
        Print a summary of option metrics along with the input parameters.

        The summary includes the option price, delta, gamma, vega, theta, and rho.
        """

        print("\n" +
              f"European {self.opt_type} summary metrics - CCR Binomial Tree\n" +
              "\n" +
              f"{'Option Price':<15} {self.option_price:.4f}\n" +
              f"{'Delta':<15} {self.delta:.4f}\n" +
              f"{'Gamma':<15} {self.gamma:.4f}\n" +
              f"{'Vega':<15} {self.vega:.4f}\n" +
              f"{'Theta':<15} {self.theta:.4f}\n" +
              f"{'Rho':<15} {self.rho:.4f}\n")

    def get_summary_metrics(self):
        """
        Re run the calculations and print a summary of option metrics along with the input parameters.

        The summary includes the option price, delta, gamma, vega, theta, and rho.
        """
        self._calculate_option_price_and_greeks()

        print("\n" +
              f"European {self.opt_type} summary metrics - CCR Binomial Tree\n" +
              "\n" +
              f"{'Option Price':<15} {self.option_price:.4f}\n" +
              f"{'Delta':<15} {self.delta:.4f}\n" +
              f"{'Gamma':<15} {self.gamma:.4f}\n" +
              f"{'Vega':<15} {self.vega:.4f}\n" +
              f"{'Theta':<15} {self.theta:.4f}\n" +
              f"{'Rho':<15} {self.rho:.4f}\n")

    def get_input_parameters(self):
        """
        Prints the input parameters.
        """

        print("\n" +
              f"European {self.opt_type} input parameters\n" +
              "\n" +
              f"{'Underlying Price':<30} {self.S0}\n" +
              f"{'Strike Price':<30} {self.K}\n" +
              f"{'Time to expiration (years)':<30} {self.T}\n" +
              f"{'Risk Free Rate':<30} {self.r}\n" +
              f"{'Volatility':<30} {self.sigma}\n" +
              f"{'Tree Sub Periods':<30} {self.N}\n")