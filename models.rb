# ======================== Logistic Regression L2 ========================= #
class LogisticRegressionModelL2
  def initialize reg_param
    @reg_param = reg_param
  end

  def predict row, w
    x = row["features"]    
    1.0 / (1 + Math.exp(-dot(w, x)))
  end
  
  def adjust w
    w.each_key {|k| w[k] = 0.0 if w[k].nan? or w[k].infinite?}
    w.each_key {|k| w[k] = 0.0 if w[k].abs > 1e5 }
  end
  
  def func data, w
    loss = data.inject(0.0) do |u,row| 
      y = row["target"].to_f > 0 ? 1.0 : -1.0
      x = row["features"]
      y_hat = dot(w,x)
      
      u += Math.log(1 + Math.exp(-y * y_hat))
    end / data.size.to_f
    
    loss + @reg_param * 0.5 * (norm(w) ** 2.0)
  end
  
  def grad data, w
    g = Hash.new {|h,k| h[k] = 0.0}
    data.each do |row| 
      y = row["target"].to_f > 0 ? 1.0 : 0.0
      x = row["features"]
      y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
      syh = 1.0 / (1 + Math.exp(-y_hat))
      x.each_key do |k|
        g[k] += (syh - y) * x[k]
      end
    end
    g.each_key {|k| g[k] = (g[k] / data.size) + @reg_param * w[k]}
    return g
  end
end

# ============================ Decision Tree ============================== #
# ================================= SVM =================================== #