// use std::iter::zip;

#[derive(Debug)]
pub struct InitSeq {
    var_pos: f64,
    var_dec: f64,
    var_con: f64,
    gamma_pos: Vec<f64>,
    gamma_dec: Vec<f64>,
    gamma_con: Vec<f64>,
}
pub enum Estimator {
    Positive,
    Monotone,
    Convex,
}
use self::Estimator::*;
impl InitSeq {
    pub fn var(&self, rhs: Estimator) -> f64 {
        match rhs {
            Positive => self.var_pos,
            Monotone => self.var_dec,
            Convex => self.var_con,
        }
    }
    pub fn gamma<'a>(&'a self, rhs: Estimator) -> &'a Vec<f64> {
        match rhs {
            Positive => &self.gamma_pos,
            Monotone => &self.gamma_dec,
            Convex => &self.gamma_con,
        }
    }
}

#[inline]
fn offset_dot_self(x: &[f64], lb: usize, ub: usize) -> f64 {
    x[..ub]
        .iter()
        .zip(x[lb..].iter())
        .map(|(a, b)| *a * *b)
        .sum::<f64>()
}

/// Compute the forward difference, *f(x + h) - f(x)*, in-place,
/// leaving the first element unchanged.
fn diff_in_place(x: &mut [f64]) {
    let mut iter = x.iter_mut().rev();
    // left, right from the perspective of the sequence taken in reverse.
    if let Some(lhs) = iter.next() {
        let mut lhs: &mut f64 = lhs;
        while let Some(rhs) = iter.next() {
            *lhs -= *rhs;
            lhs = rhs;
        }
    }
}

pub fn init_seq(x: &[f64]) -> InitSeq {
    let n = x.len();
    let inv_n = 1.0 / n as f64;
    let mu = x.iter().sum::<f64>() * inv_n;
    let z: Vec<f64> = x.iter().map(move |x_i| *x_i - mu).collect();

    let mut gamma_hat: Vec<f64> = vec![0.0; n / 2];
    let mut gamma_0: f64 = 0.0;
    let mut k: usize = 0;

    for g_hat in gamma_hat.iter_mut() {
        let lb = 2 * k;
        let ub = n - lb;
        let gamma_2k = offset_dot_self(&z, lb, ub) * inv_n;
        if k == 0 {
            gamma_0 = gamma_2k;
        }

        let lb = lb + 1;
        let ub = ub - 1;
        let gamma_2k1 = offset_dot_self(&z, lb, ub) * inv_n;

        let gamma_hat_k = gamma_2k + gamma_2k1;
        if gamma_hat_k > 0.0 {
            *g_hat = gamma_hat_k;
            k += 1;
        } else {
            *g_hat = 0.0;
            k += 1;
            break;
        }
    }
    gamma_hat.truncate(k);

    // let mut gamma_pos: Vec<f64> = vec![0.0; k];
    // let mut gamma_dec: Vec<f64> = vec![0.0; k];
    // let mut min = f64::MAX;
    // for (hat, (pos, dec)) in zip(
    //     gamma_hat.iter_mut(),
    //     zip(gamma_pos.iter_mut(), gamma_dec.iter_mut()),
    // ) {
    //     *pos = *hat;
    //     min = hat.min(min);
    //     *hat = min;
    //     *dec = min;
    // }
    let mut min = f64::MAX;
    let (gamma_pos, gamma_dec): (Vec<_>, Vec<_>) = gamma_hat
        .iter_mut()
        .map(move |hat| {
            let pos = *hat;
            min = hat.min(min);
            *hat = min;
            (pos, min)
        })
        .unzip();

    // Greatest convex minorant via isotonic regression on derivative
    diff_in_place(gamma_hat.as_mut_slice());

    let v = &gamma_hat[1..];
    let n = v.len();
    let mut nu: Vec<f64> = Vec::with_capacity(n);
    nu.push(v[0]);
    let mut w: Vec<usize> = Vec::with_capacity(n);
    w.push(1);
    let mut j: usize = 0;
    let mut i: usize = 1;
    while i < n {
        j += 1;
        nu.push(v[i]);
        w.push(1);
        i += 1;
        while j > 0 && nu[j - 1] > nu[j] {
            let w_prime = w[j - 1] + w[j];
            let nu_prime = (w[j - 1] as f64 * nu[j - 1] + w[j] as f64 * nu[j]) / w_prime as f64;
            nu[j - 1] = nu_prime;
            w[j - 1] = w_prime;
            nu.pop();
            w.pop();
            j -= 1;
        }
    }
    let mut gamma_con = gamma_hat;
    let mut pos: usize = 1;
    let mut nu_prev: f64 = gamma_con[0];
    for (nu_j, w_j) in nu.into_iter().zip(w.into_iter()) {
        for mu_pos in gamma_con[pos..pos + w_j].iter_mut() {
            *mu_pos = nu_prev + nu_j;
            nu_prev = mu_pos.clone();
        }
        pos += w_j;
    }

    let var_pos = 2.0 * gamma_pos.iter().sum::<f64>() - gamma_0;
    let var_dec = 2.0 * gamma_dec.iter().sum::<f64>() - gamma_0;
    let var_con = 2.0 * gamma_con.iter().sum::<f64>() - gamma_0;

    InitSeq {
        var_pos,
        var_dec,
        var_con,
        gamma_pos,
        gamma_dec,
        gamma_con,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_in_place_works() {
        let mut x: Vec<f64> = vec![3.0, 4.0, 1.0, 5.5];
        diff_in_place(x.as_mut_slice());
        assert_eq!(x, vec![3.0, 1.0, -3.0, 4.5]);

        let mut x: Vec<f64> = vec![1.0, 3.0];
        diff_in_place(x.as_mut_slice());
        assert_eq!(x, vec![1.0, 2.0]);

        let mut x: Vec<f64> = vec![1.0];
        diff_in_place(x.as_mut_slice());
        assert_eq!(x, vec![1.0]);

        let mut x: Vec<f64> = vec![];
        diff_in_place(x.as_mut_slice());
        assert_eq!(x, vec![]);
    }
}
