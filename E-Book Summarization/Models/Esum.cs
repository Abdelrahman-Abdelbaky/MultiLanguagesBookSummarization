namespace GP.FullProject.Models
{
    public class Esum
    {
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; }
        public string sum { get; set; }
        public int EBookId { get; set; }
        public EBook EBook { get; set; }
        
    }
}
