


namespace GP.FullProject.Models
{
    public class EBook
    {
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; }
        [MaxLength(100)]
        public string BookName { get; set; }
        public float percentage { get; set; }
        public byte[] Content { get; set; }

    }
}
